"""Closed imports of the previously executed native graph; never load a model."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

HERE = Path(__file__).resolve().parent


def load():
    pins = json.loads((HERE / "NATIVE-SOURCE-PINS.json").read_text())
    for row in pins["files"]:
        raw = (HERE / row["path"]).read_bytes()
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("unchanged_executed_native_source_required")
    directory = HERE / "retained/native"
    aliases = {name: sys.modules.get(name)
               for name in ("candidate_query", "input_contract", "native_worker")}
    loaded = {}
    try:
        for name in aliases:
            spec = importlib.util.spec_from_file_location(name, directory / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
        loaded["input_contract"].hasher()
    finally:
        for name, old in aliases.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return loaded
