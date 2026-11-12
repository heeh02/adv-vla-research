import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import ml_dtypes
import native_codec as C


class Tests(unittest.TestCase):
    def test_native_bits_zero_sign_scalar_shape_and_structure_survive(self):
        bits = np.asarray([0, 0x8000, 0x3f80, 0xbf81, 1, 0x7f7f], dtype=np.uint16)
        value = {'bf': bits.view(ml_dtypes.bfloat16),
                 'fp': np.asarray([-0., 0.3999999165534973], np.float32),
                 'flag': np.asarray(True), 'nested': (None, [1, 'literal', np.asarray(10, np.int32)])}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'call.npz'; ref = C.save_tree(p, value); got = C.load_tree(p, ref)
            self.assertEqual(got['bf'].tobytes(), bits.tobytes())
            self.assertEqual(got['bf'].dtype.name, 'bfloat16')
            self.assertEqual(got['fp'].tobytes(), value['fp'].tobytes())
            self.assertEqual(got['flag'].shape, ())
            self.assertIsInstance(got['nested'], tuple)
            self.assertEqual(got['nested'][1][2].dtype, np.int32)
            with self.assertRaises(FileExistsError): C.save_tree(p, value)

    def test_wholefile_and_leaf_corruption_are_independently_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'call.npz'; ref = C.save_tree(p, {'x': np.arange(3, dtype=np.float32)})
            with self.assertRaisesRegex(ValueError, 'wholefile'):
                C.load_tree(p, {**ref, 'sha256': '0'*64})
            with np.load(p, allow_pickle=False) as z: arrays = {k:z[k].copy() for k in z.files}
            arrays['a_000000'][0] = 2
            q = Path(tmp)/'tampered.npz'; np.savez_compressed(q, **arrays)
            with self.assertRaisesRegex(ValueError, 'leaf_SHA'):
                C.load_tree(q, C.file_ref(q))

    def test_nonfinite_object_and_extra_member_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for a in (np.asarray([float('nan')]), np.asarray([object()])):
                with self.assertRaises(ValueError): C.save_tree(Path(tmp)/'bad.npz', a)
            p = Path(tmp)/'call.npz'; C.save_tree(p, np.arange(2, dtype=np.int32))
            with np.load(p, allow_pickle=False) as z: arrays = {k:z[k].copy() for k in z.files}
            arrays['unexpected'] = np.asarray([0], np.int32)
            q = Path(tmp)/'extra.npz'; np.savez_compressed(q, **arrays)
            with self.assertRaisesRegex(ValueError, 'complete_roster'): C.load_tree(q, C.file_ref(q))

    def test_explicit_failed_call_snapshot_retains_nonfinite_bits_without_qualification(self):
        bits = np.asarray([0x7f80, 0xff80, 0x7fc1, 0x8000], dtype=np.uint16)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'failed.npz'
            ref = C.save_tree(p, bits.view(ml_dtypes.bfloat16), allow_nonfinite=True)
            with self.assertRaisesRegex(ValueError, 'finite_dtype'): C.load_tree(p, ref)
            restored = C.load_tree(p, ref, allow_nonfinite=True)
            self.assertEqual(restored.view(np.uint16).tobytes(), bits.tobytes())


if __name__ == '__main__': unittest.main()
