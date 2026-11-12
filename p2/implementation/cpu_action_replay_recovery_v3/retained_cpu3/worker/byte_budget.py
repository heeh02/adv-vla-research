"""Scoped HTTP-body budget for Q1 urllib readers AND http.client BOSWriter.

Usage::

    with bounded_http(total_limit_bytes=8*1024**3,
                      input_limit_bytes=6*1024**3,
                      output_limit_bytes=256*1024**2,
                      initial_input_bytes=admitted_package_bytes) as budget:
        with budget.download_bound(original_metadata_limit):
            original_get_small(...)  # optional tighter reservation; no request edit
        # Join all science workers before entering this irreversible phase.
        with budget.archive_phase():
            original_writer.put(partial_result, relative)
        counts = budget.snapshot()
        # A complete result requires rejected_count == 0, plus scientific gates.

Reservations happen before HTTP request/body transmission. Successful fully-read
requests settle to observed body bytes. Failed/incomplete requests retain their
FULL reservation, including error-body allowance; they never refund traffic.
Upload observations mean bytes supplied to HTTP, not server acknowledgements.
Header/TLS/TCP overhead is outside this APPLICATION-body budget. Unknown response
length, chunked/compressed responses, unknown upload length and redirects are not
admitted. No URL, object key, header, credential or response body enters snapshots
or exceptions. Opaque object identities exist only in memory for archive GETs.

The process-wide hook must enclose all worker threads. Do not nest installations
or use it concurrently with an unrelated HTTP workload. No network is opened by
installation. Local cache/output-file size limits remain the worker's job.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import http.client
import re
import threading

GIB = 1024 ** 3
MIB = 1024 ** 2
CHUNK_BYTES = 64 * 1024
_INSTALL_LOCK = threading.Lock()
_DOWNLOAD_BOUND = ContextVar("generation_geometry_download_bound", default=None)
_MISSING = object()


class ByteBudgetError(RuntimeError):
    """Sanitized policy/transport failure. The message is a fixed reason code."""


def _integer(value, *, positive=False):
    return type(value) is int and value >= (1 if positive else 0)


class _Ticket:
    def __init__(self, budget, incoming, outgoing, science, key, method):
        self.budget = budget
        self.incoming, self.outgoing, self.science = incoming, outgoing, science
        self.key, self.method = key, method
        self.downloaded = self.supplied = 0
        self.declared = self.status = None
        self.failed = self.finished = False
        self.close_response = None

    def observe(self, count, *, upload=False):
        with self.budget._lock:
            if self.finished:
                raise ByteBudgetError("request_already_closed")
            if upload:
                self.supplied += count
                self.budget._observed_upload += count
                good = self.supplied <= self.outgoing
            else:
                self.downloaded += count
                self.budget._observed_download += count
                good = self.downloaded <= self.incoming
            if not good:
                self.failed = True
                self.budget._reject("transport_exceeded_reserved_body")

    def finish(self, *, failed=False):
        with self.budget._lock:
            if self.finished:
                return
            success = (not failed and not self.failed and self.status is not None
                       and 200 <= self.status < 300 and self.declared is not None
                       and self.downloaded == self.declared and self.supplied == self.outgoing)
            incoming = self.downloaded if success else self.incoming
            outgoing = self.supplied if success else self.outgoing
            b = self.budget
            b._reserved_input -= self.incoming
            b._reserved_output -= self.outgoing
            if self.science:
                b._reserved_science -= self.incoming
                b._science_charged += incoming
            else:
                b._archive_charged += incoming
            b._input_charged += incoming
            b._output_charged += outgoing
            if success:
                b._succeeded += 1
            else:
                b._failed += 1
            self.finished = True
            b._tickets.discard(self)

    def transport_failure(self, reason):
        self.failed = True
        with self.budget._lock:
            self.budget._errors[reason] += 1
        self.finish(failed=True)


class _Upload:
    def __init__(self, raw, ticket):
        self._raw, self._ticket = raw, ticket

    def read(self, size=CHUNK_BYTES):
        remaining = self._ticket.outgoing - self._ticket.supplied
        if remaining == 0:
            return b""
        take = min(CHUNK_BYTES, remaining, size if isinstance(size, int) and size > 0 else CHUNK_BYTES)
        try:
            data = self._raw.read(take)
        except Exception:
            raise ByteBudgetError("upload_source_read_failed") from None
        if not isinstance(data, bytes) or not data or len(data) > take:
            raise ByteBudgetError("upload_source_length_or_type_invalid")
        self._ticket.observe(len(data), upload=True)
        return data


class _Response:
    def __init__(self, raw, ticket):
        self._raw, self._ticket, self._closed = raw, ticket, False
        ticket.close_response = self.close
        status = getattr(raw, "status", None)
        if type(status) is not int:
            ticket.budget._reject("invalid_response_status")
        ticket.status = status
        if 300 <= status < 400:
            ticket.budget._reject("redirect_response_not_admitted")
        length = raw.getheader("Content-Length")
        if length is None and status in (204, 304):
            length = "0"  # protocol-defined empty body, not an inferred length
        if not isinstance(length, str) or not re.fullmatch(r"[0-9]+", length):
            ticket.budget._reject("unknown_or_invalid_response_length")
        declared = int(length)
        if declared > ticket.incoming:
            ticket.budget._reject("response_exceeds_reserved_bound")
        if raw.getheader("Transfer-Encoding") not in (None, "identity"):
            ticket.budget._reject("chunked_response_not_admitted")
        if raw.getheader("Content-Encoding") not in (None, "identity"):
            ticket.budget._reject("encoded_response_not_admitted")
        ticket.declared = declared
        # A PUT acknowledged or explicitly reconciled by BOSWriter permits only
        # the same opaque object to be read in the archive phase.
        if ticket.method == "PUT" and (200 <= status < 300 or status in (409, 412)):
            with ticket.budget._lock:
                ticket.budget._put_sizes[ticket.key] = ticket.outgoing

    def __getattr__(self, name):
        if name in ("status", "code", "reason", "headers", "version", "will_close", "chunked"):
            return getattr(self._raw, name)
        raise AttributeError(name)  # do not expose raw .fp or unmetered readers

    @property
    def closed(self):
        return self._closed

    def getheader(self, name, default=None):
        return self._raw.getheader(name, default)

    def getheaders(self):
        return self._raw.getheaders()

    def info(self):
        return self._raw.headers

    def _one(self, size):
        t = self._ticket
        if self._closed or t.finished:
            raise ByteBudgetError("response_already_closed")
        remaining = t.declared - t.downloaded
        if size == 0 or remaining == 0:
            return b""
        take = min(CHUNK_BYTES, size, remaining)
        try:
            data = self._raw.read1(take)
        except Exception:
            t.transport_failure("response_read_failed")
            raise ByteBudgetError("response_read_failed") from None
        if not isinstance(data, bytes) or len(data) > take:
            t.transport_failure("invalid_response_read")
            raise ByteBudgetError("invalid_response_read")
        if not data:
            t.transport_failure("truncated_response")
            raise ByteBudgetError("truncated_response")
        t.observe(len(data))
        return data

    def read1(self, size=-1):
        return self._one(CHUNK_BYTES if size is None or size < 0 else size)

    def read(self, size=-1):
        remaining = self._ticket.declared - self._ticket.downloaded
        wanted = remaining if size is None or size < 0 else min(size, remaining)
        chunks = []
        while wanted:
            data = self._one(wanted)
            chunks.append(data)
            wanted -= len(data)
        return b"".join(chunks)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self._raw.close()
            except Exception:
                self._ticket.failed = True
                with self._ticket.budget._lock:
                    self._ticket.budget._errors["response_close_failed"] += 1
            finally:
                self._ticket.finish()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, tb):
        if kind is not None:
            self._ticket.failed = True
        self.close()
        return False


class bounded_http:
    def __init__(self, *, total_limit_bytes=8*GIB, input_limit_bytes=6*GIB,
                 output_limit_bytes=256*MIB, initial_input_bytes=0,
                 initial_output_bytes=0, max_get_bytes=64*MIB,
                 error_response_bound=8192):
        values = (total_limit_bytes, input_limit_bytes, output_limit_bytes,
                  initial_input_bytes, initial_output_bytes, max_get_bytes,
                  error_response_bound)
        if not all(_integer(v) for v in values) or max_get_bytes == 0:
            raise ByteBudgetError("invalid_budget_configuration")
        if (initial_input_bytes > input_limit_bytes or initial_output_bytes > output_limit_bytes
                or initial_input_bytes + initial_output_bytes > total_limit_bytes):
            raise ByteBudgetError("bootstrap_already_exceeds_budget")
        self.total_limit_bytes = total_limit_bytes
        self.input_limit_bytes, self.output_limit_bytes = input_limit_bytes, output_limit_bytes
        self.max_get_bytes, self.error_response_bound = max_get_bytes, error_response_bound
        self._lock = threading.RLock()
        self._input_charged = self._science_charged = initial_input_bytes
        self._output_charged = initial_output_bytes
        self._archive_charged = 0
        self._reserved_input = self._reserved_output = self._reserved_science = 0
        self._observed_download = self._observed_upload = 0
        self._started = self._succeeded = self._failed = self._rejected = 0
        self._errors = Counter()
        self._tickets, self._put_sizes = set(), {}
        self._archive = self._science_blocked = self._active = False
        self._originals = None

    def _reject(self, reason):
        with self._lock:
            self._rejected += 1
            self._errors[reason] += 1
            self._science_blocked = True
        raise ByteBudgetError(reason) from None

    @contextmanager
    def download_bound(self, limit):
        if not _integer(limit):
            raise ByteBudgetError("invalid_download_bound")
        token = _DOWNLOAD_BOUND.set((self, limit))
        try:
            yield
        finally:
            _DOWNLOAD_BOUND.reset(token)

    @contextmanager
    def archive_phase(self):
        with self._lock:
            if not self._active:
                self._reject("budget_not_active")
            if self._tickets:
                self._reject("join_science_requests_before_archive")
            self._archive = True
            self._science_blocked = True
        yield self  # irreversible: new science downloads never resume

    def _prepare(self, conn, method, path, body, headers, encode_chunked):
        with self._lock:
            if not self._active:
                self._reject("budget_not_active")
            if self._science_blocked and not self._archive:
                self._reject("science_admission_stopped")
            if method not in ("GET", "PUT") or encode_chunked:
                self._reject("unbounded_request_method_or_encoding")
            hh = {str(k).lower(): v for k, v in headers.items()}
            if len(hh) != len(headers) or hh.get("transfer-encoding") is not None:
                self._reject("ambiguous_request_headers")
            length = hh.get("content-length")
            if length is None and body is None:
                outgoing = 0
            elif isinstance(length, str) and re.fullmatch(r"[0-9]+", length):
                outgoing = int(length)
            else:
                self._reject("unknown_upload_length")
            if body is None and outgoing != 0:
                self._reject("upload_body_missing")
            if method == "GET" and (outgoing or body is not None):
                self._reject("GET_body_not_admitted")
            if body is not None and not isinstance(body, (bytes, bytearray)) and not callable(getattr(body, "read", None)):
                self._reject("unknown_upload_stream_type")
            if isinstance(body, (bytes, bytearray)) and len(body) != outgoing:
                self._reject("upload_content_length_mismatch")
            # Opaque identifier is never exported. No signed headers are retained.
            key = hashlib.sha256(repr((conn.host, conn.port, path)).encode()).digest()
            if method == "GET":
                if self._archive and key not in self._put_sizes:
                    self._reject("archive_GET_requires_registered_PUT")
                byte_range = hh.get("range")
                if byte_range is not None:
                    match = re.fullmatch(r"bytes=([0-9]+)-([0-9]+)", str(byte_range))
                    if match is None or int(match[2]) < int(match[1]):
                        self._reject("unknown_Range_length")
                    bound = int(match[2]) - int(match[1]) + 1
                else:
                    hint = _DOWNLOAD_BOUND.get()
                    bound = hint[1] if hint is not None and hint[0] is self else self.max_get_bytes
                    if key in self._put_sizes:
                        bound = min(bound, self._put_sizes[key])
                incoming = max(bound, self.error_response_bound)
            else:
                incoming = self.error_response_bound
            total = self._input_charged + self._output_charged + self._reserved_input + self._reserved_output
            if total + incoming + outgoing > self.total_limit_bytes:
                self._reject("total_byte_limit")
            if self._output_charged + self._reserved_output + outgoing > self.output_limit_bytes:
                self._reject("output_byte_limit")
            if not self._archive and self._science_charged + self._reserved_science + incoming > self.input_limit_bytes:
                self._reject("science_input_byte_limit")
            ticket = _Ticket(self, incoming, outgoing, not self._archive, key, method)
            self._reserved_input += incoming
            self._reserved_output += outgoing
            if ticket.science:
                self._reserved_science += incoming
            self._tickets.add(ticket)
            self._started += 1
            return ticket

    def _connection_methods(self, originals):
        budget = self

        def close_raw(raw):
            if raw is not None:
                try:
                    raw.close()
                except Exception:
                    with budget._lock:
                        budget._errors["response_close_failed"] += 1

        class Methods:
            def request(self, method, url, body=None, headers=None, *, encode_chunked=False):
                previous = getattr(self, "_byte_ticket", None)
                if previous is not None and not previous.finished:
                    budget._reject("previous_request_not_closed")
                ticket = budget._prepare(self, method, url, body, headers or {}, encode_chunked)
                self._byte_ticket = ticket
                supplied = body
                if isinstance(body, (bytes, bytearray)):
                    ticket.observe(len(body), upload=True)
                elif body is not None:
                    supplied = _Upload(body, ticket)
                try:
                    return originals["request"](self, method, url, body=supplied, headers=headers or {}, encode_chunked=encode_chunked)
                except Exception:
                    ticket.transport_failure("request_transport_failed")
                    raise ByteBudgetError("request_transport_failed") from None

            def getresponse(self, *args, **kwargs):
                ticket = getattr(self, "_byte_ticket", None)
                if ticket is None or ticket.finished:
                    budget._reject("response_without_active_reservation")
                raw = None
                try:
                    # HTTPConnection.getresponse may call self.close() when the
                    # server says Connection: close, while the response body is
                    # still valid through its file object. Do not settle yet.
                    self._byte_receiving_headers = True
                    raw = originals["getresponse"](self, *args, **kwargs)
                    return _Response(raw, ticket)
                except ByteBudgetError:
                    close_raw(raw)
                    ticket.finish(failed=True)
                    raise
                except Exception:
                    close_raw(raw)
                    ticket.transport_failure("response_headers_failed")
                    raise ByteBudgetError("response_headers_failed") from None
                finally:
                    self._byte_receiving_headers = False

            def close(self):
                ticket = getattr(self, "_byte_ticket", None)
                try:
                    return originals["close"](self)
                except Exception:
                    if ticket is not None:
                        ticket.transport_failure("connection_close_failed")
                    raise ByteBudgetError("connection_close_failed") from None
                finally:
                    if ticket is not None and not getattr(self, "_byte_receiving_headers", False):
                        ticket.finish()

        return {name: getattr(Methods, name) for name in originals}

    def snapshot(self):
        with self._lock:
            total = self._input_charged + self._output_charged
            return {
                "total_limit_bytes": self.total_limit_bytes,
                "science_input_limit_bytes": self.input_limit_bytes,
                "output_limit_bytes": self.output_limit_bytes,
                "charged_total_bytes": total, "charged_input_bytes": self._input_charged,
                "charged_science_input_bytes": self._science_charged,
                "charged_archive_input_bytes": self._archive_charged,
                "charged_output_bytes": self._output_charged,
                "reserved_input_bytes": self._reserved_input,
                "reserved_output_bytes": self._reserved_output,
                "observed_download_bytes": self._observed_download,
                "upload_bytes_supplied": self._observed_upload,
                "remaining_total_admission_bytes": self.total_limit_bytes - total - self._reserved_input - self._reserved_output,
                "requests_started": self._started, "requests_succeeded": self._succeeded,
                "requests_failed": self._failed, "rejected_count": self._rejected,
                "inflight_requests": len(self._tickets), "archive_phase": self._archive,
                "science_admission_stopped": self._science_blocked,
                "errors": dict(self._errors),
            }

    def __enter__(self):
        if not _INSTALL_LOCK.acquire(blocking=False):
            raise ByteBudgetError("another_budget_hook_is_active")
        try:
            if self._originals is not None:
                raise ByteBudgetError("budget_context_cannot_be_reused")
            # Keep class identities: Python 3.9 HTTPSConnection.__init__ uses
            # super(HTTPSConnection, self), so replacing its global class name
            # with a subclass recurses. Capture ALL methods before any patch,
            # including HTTPS methods inherited from HTTPConnection.
            names = ("request", "getresponse", "close")
            self._originals = []
            plans = []
            for cls in dict.fromkeys((http.client.HTTPConnection, http.client.HTTPSConnection)):
                saved = {name: cls.__dict__.get(name, _MISSING) for name in names}
                methods = {name: getattr(cls, name) for name in names}
                self._originals.append((cls, saved))
                plans.append((cls, self._connection_methods(methods)))
            self._active = True
            for cls, methods in plans:
                for name, method in methods.items():
                    setattr(cls, name, method)
            return self
        except Exception:
            _INSTALL_LOCK.release()
            raise

    def __exit__(self, kind, value, tb):
        try:
            with self._lock:
                self._active = False
                active = list(self._tickets)
            for ticket in active:
                ticket.failed = True
                if ticket.close_response is not None:
                    ticket.close_response()
                ticket.finish(failed=True)
        finally:
            for cls, methods in self._originals:
                for name, method in methods.items():
                    if method is _MISSING:
                        delattr(cls, name)
                    else:
                        setattr(cls, name, method)
            self._put_sizes.clear()
            _INSTALL_LOCK.release()
        return False
