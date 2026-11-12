"""Synthetic-only pair math and old native qualification regressions."""
from pathlib import Path
import copy
import hashlib
import json
import time
import unittest
from unittest.mock import patch
import numpy as np
import adapter as A
import pair as P
import quantization_frozen as Q
OLD=A.module('_pair_test_frozen_precision',A.HERE/'frozen_precision/adapter.py')

def fixture():
    N, M = OLD.load_helpers()
    def native(value, logical=None):
        a = np.asarray(value)
        desc = {"shape": list(a.shape), "logical_dtype": logical or a.dtype.name,
                "storage_dtype": str(a.dtype), "bytes": a.nbytes,
                "payload_sha256": hashlib.sha256(a.tobytes()).hexdigest()}
        return N.Native(a, desc)
    def wrap(value):
        if isinstance(value, dict):
            return {k: wrap(v) for k, v in value.items()}
        return native(value.storage, value.dtype) if isinstance(value, N.Native) else native(value)
    def bf(shape, bits=0):
        return native(np.full(shape, bits, np.uint16), "bfloat16")
    n = 16
    ref = {"path": "000002-PVP_INPUT_VERIFIED.npz", "bytes": 1, "sha256": "a" * 64, "uri": "synthetic:input"}
    inp = N.Record("PVP_INPUT_VERIFIED", {"root_id": OLD.ROOT, "query_index": 11, "sources": {"synthetic": True}},
        wrap({"noise": np.zeros((1, 10, 32), np.float32), "consumed_decode_state": np.zeros((1, 32), np.float32),
              "Vmod": bf((1, n, 1, 256), 0x3f00), "actual_E": np.full((1, n, 1, 256), .25, np.float32)}), ref, 1, True)
    calls, entries = [], []
    for ri, (role, call_id) in enumerate(zip(OLD.ROLES, OLD.CALL_IDS)):
        entry = {"call_id": call_id, "root_id": OLD.ROOT, "query_index": 11, "role": role, "receiver": "T", "donor": "C",
                 "num_steps": 10, "kv_mask": [False, ri > 0], "layer_mask": [ri > 0 and j == 8 for j in range(18)],
                 "flow_mask": [ri == 1 and j == 5 or ri == 2 and j >= 5 for j in range(10)], "use_vmod": ri > 0}
        entries.append(entry)
        reads = {k: bf((5, 3, 1, 10, 1024), 0x3f80 if ri else 0x4000) for k in M.FIELDS}
        if ri == 2:
            reads["attention_output_post_WO_pre_gate"].storage[1, 1] = 0x3f81
        reads.update(query_post_rope_scaled=bf((5, 3, 1, 10, 1, 8, 256)),
            probabilities_native=bf((5, 3, 1, 1, 8, 10, n + 10)), attention_gate=bf((5, 3, 1, 1, 1024), 0x3f00),
            suffix_key_post_rope=bf((5, 3, 1, 10, 1, 256)), suffix_value=bf((5, 3, 1, 10, 1, 256)),
            mask=np.ones((5, 1, 10, n + 10), bool), suffix_positions=np.broadcast_to(np.arange(n, n + 10, dtype=np.int32), (5, 1, 10)).copy(),
            query_defined=np.ones((5, 1, 10), bool), adarms_condition=np.zeros((5, 1, 1024), np.float32))
        x = np.zeros((11, 1, 10, 32), np.float32)
        velocity = bf((10, 1, 10, 32))
        if ri:
            x[6:] = .25
            velocity.storage[5:] = 0x3f80
        if ri == 2:
            x[7:] = .5
            velocity.storage[6:] = 0x4000
        arrays = wrap({"reads": reads, "flow": {"x": x, "velocity_native": velocity,
            "time": np.linspace(1, 0, 11, dtype=np.float32), "time_weak_type": np.ones(11, bool), "dt": np.full(10, -.1, np.float32)},
            "prefix_mask": np.ones((1, n), bool), "prefix_positions": np.arange(n, dtype=np.int32)[None],
            "selectors": {k: np.asarray(entry[k], bool) for k in ("kv_mask", "layer_mask", "flow_mask", "use_vmod")},
            "capture_layers": np.array(M.LAYERS, np.int32), "capture_calls": np.array(M.FLOWS, np.int32),
            "gates": {k: np.asarray(True) for k in M.GATES}, "raw32": x[-1], "adapter7": np.zeros((1, 10, 7), np.float64)})
        meta = {**entry, "graph": "Gstar", "input_ref": ref, "input_identity": "b" * 64,
                "PG_refs": {"T": {"path": "syntheticPG", "bytes": 1, "sha256": "c" * 64}},
                "Vmod_actual_identity": {"shape": [1, n, 1, 256], "logical_dtype": "bfloat16",
                    "bytes": inp.field("Vmod").storage.nbytes, "sha256": N.sha(inp.field("Vmod").storage.tobytes())}}
        calls.append(N.Record("PVP_CALL_OUTPUT", meta, arrays,
                              {"path": f"{ri+10:06d}-PVP_CALL_OUTPUT.npz", "bytes": 1, "sha256": str(ri) * 64}, 1, True))
    source = {"old_saved_sources": {"synthetic": True}, "prefix_capacity": n, "valid_prefix_tokens": n, "exactzero_actual_E": False}
    return inp, calls[1:], entries[1:], source, N, M


def math_record(w=0x3f80,g=0x3f00,r=0x4480,y=0x4480):
    return {'W_used':np.full((10,1024),w,np.uint16),'gate':np.full((1,1024),g,np.uint16),
            'residual_input':np.full((10,1024),r,np.uint16),'after_attention_residual':np.full((10,1024),y,np.uint16)}


class Tests(unittest.TestCase):
    def test_pair_exactly_reuses_frozen_natural_d_metrics(self):
        p,s=math_record(),math_record(w=0x3f81)
        result=P.analyze_pair(p,s)
        synthetic_c=math_record(w=0x3f82)
        def arm(record):return {**record,'W_before':p['W_used']}
        old=Q.analyze_native_triplet({'none':arm(p),'natural_d':arm(s),'common_c':arm(synthetic_c)})
        reference=[x for x in old['rows']if x['contrast']=='d_minus_none']
        self.assertEqual(len(result['rows']),13)
        for a,b in zip(result['rows'],reference):
            self.assertEqual({k:v for k,v in a.items()if k!='contrast'},{k:v for k,v in b.items()if k!='contrast'})
        self.assertEqual([r['role']for r in result['rounding_candidates']['records']],['archived_pulse','archived_persistent'])
        self.assertFalse(result['common_c_record_created'])
        self.assertLess(len(A.encode(result)),A.LIMIT)

    def test_absorbed_increment_signed_cross_and_zero_denominators_retained(self):
        r=P.analyze_pair(math_record(),math_record(w=0x3f81))
        full=next(x for x in r['rows']if x['window']=='full10')
        self.assertEqual(full['ideal_nonzero_but_native_zero_coordinates'],10240)
        self.assertEqual(full['metrics']['native_after_attention_increment']['energy'],0.)
        self.assertLess(full['ideal_error_signed_cross'],0.)
        self.assertIsNone(full['combined_error_over_actual_energy'])
        zero=P.analyze_pair(math_record(g=0),math_record(w=0x3f81,g=0))
        self.assertTrue(all(x['combined_error_over_ideal_energy']is None for x in zero['rows']))

    def test_two_actual_roles_and_source_receiver_gates(self):
        args=fixture();proof=A.validate_pair(*args,synthetic=True)
        self.assertTrue(proof['two_CALLs_only'])
        self.assertEqual(len(args[1]),2)
        for path,index in [(('flow','x'),(6,0,0,0)),(('flow','time_weak_type'),(6,)),
                           (('reads','query_post_rope_scaled'),(1,1,0,0,0,0,0)),
                           (('reads','probabilities_native'),(1,1,0,0,0,0,0)),
                           (('reads','receiver_residual_input'),(1,1,0,0,0)),
                           (('reads','attention_gate'),(1,1,0,0,0)),(('reads','mask'),(1,0,0,0))]:
            with self.subTest(path=path):
                changed=fixture();value=changed[1][1].field(*path).storage
                value[index]=0 if value[index]else 1
                with self.assertRaises(ValueError):A.validate_pair(*changed,synthetic=True)
        changed=fixture();changed[1][1].metadata['role']='baseline_none_1'
        with self.assertRaises(ValueError):A.validate_pair(*changed,synthetic=True)

    def test_same_gate_and_residual_mandatory_no_native_product_split(self):
        p,s=math_record(),math_record(w=0x3f81)
        s['residual_input'][0,0]+=1
        with self.assertRaisesRegex(ValueError,'same_archived'):P.analyze_pair(p,s)
        p,s=math_record(),math_record(w=0x3f81,g=0x3f01)
        with self.assertRaisesRegex(ValueError,'same_archived'):P.analyze_pair(p,s)

    def test_production_Mac_gate_precedes_any_native_load(self):
        with patch.object(A.sys,'platform','darwin'),patch.object(A,'verify_sources',side_effect=AssertionError('no source/array read')):
            with self.assertRaisesRegex(ValueError,'cloud_Linux'):
                A.run_stage('/not-read','/not-written',source_manifest_sha256='a'*64,deadline_epoch=time.time()+1)

    def test_exact_five_source_subset_and_closed_manifest(self):
        inputs=A.load_inputs();original=OLD.load_inputs();by={r['role']:r for r in original['input_objects']}
        self.assertEqual(inputs['input_objects'],[by[r]for r in A.ROLES])
        self.assertEqual(sum(r['bytes']for r in inputs['input_objects']),6314847)
        self.assertNotIn('baseline_none_1',[r['role']for r in inputs['input_objects']])
        raw=(A.HERE/'SOURCE-MANIFEST.json').read_bytes()
        self.assertEqual(A.verify_sources(hashlib.sha256(raw).hexdigest()),A.identity(raw))


if __name__=='__main__':unittest.main()
