"""Synthetic semantic tests; no archived scientific rows are loaded."""
from pathlib import Path
import copy
import tempfile
import unittest
from unittest.mock import patch
import analysis as A
import adapter
from synthetic_fixture import fixture,stage_fixture


class ReadoutTests(unittest.TestCase):
    def test_complete_fixed_support(self):
        data=A.aggregate(*fixture(),synthetic=True)
        self.assertEqual(len(data['support']),1000)
        self.assertEqual(len(data['endpoints']),848)
        self.assertEqual(len(data['summary']['endpoint']),106)
        self.assertEqual(sum(r['typed_exact'] for r in data['fields'].values()),64)
        self.assertEqual(sum(r['typed_exact'] for r in data['features'].values()),40)

    def test_zero_undefined_near_one_not_reclassified(self):
        s=A.summary([0.,None,.5,1-4e-14,1.,2.,3.,4.])
        self.assertEqual((s['zero'],s['undefined'],s['less_than_one'],s['equal_to_one'],s['greater_than_one']),
                         (1,1,3,1,3))
        self.assertEqual(len(s['near_one_display_only']),2)
        self.assertEqual(s['near_one_display_only'][0]['unrounded_value'],1-4e-14)

    def test_missing_duplicate_wrong_identity(self):
        for table in range(3):
            args=list(fixture());args[table].pop()
            with self.assertRaises(ValueError):A.aggregate(*args,synthetic=True)
            args=list(fixture());args[table].append(copy.deepcopy(args[table][0]))
            with self.assertRaisesRegex(ValueError,'no_duplicate_support'):A.aggregate(*args,synthetic=True)
        args=list(fixture());args[0][0]['root_id']='wrong'
        with self.assertRaisesRegex(ValueError,'fixed8_snapshot_identity'):A.aggregate(*args,synthetic=True)

    def test_corrupt_ratio_and_zero(self):
        args=list(fixture());args[0][1]['relative_RMS_to_reference']=1.
        with self.assertRaisesRegex(ValueError,'cached_ratio_arithmetic'):A.aggregate(*args,synthetic=True)
        args=list(fixture())
        row=next(r for r in args[2] if r['comparison']=='persistent_effect_own_baselines')
        row['residual_RMS_over_new_S_RMS']=0.
        with self.assertRaisesRegex(ValueError,'undefined_iff_zero_denominator'):A.aggregate(*args,synthetic=True)

    def test_synthetic_full_SHA_archive_and_producer_binding(self):
        with tempfile.TemporaryDirectory() as d:
            stage=Path(d)/'fake';doc=stage_fixture(stage)
            data=adapter.load(stage,doc,synthetic=True)
            self.assertEqual(len(data['support']),1000)
            wrong=copy.deepcopy(doc);wrong['input_objects'][0]['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'input_whole_SHA'):adapter.load(stage,wrong,synthetic=True)

    def test_producer_binding_not_inferred_from_archive(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            stage=Path(d)/'fake';doc=stage_fixture(stage)
            name='dev16-feature-bridge-result.json'
            producer=json.loads((stage/name).read_bytes());producer['outputs'][0]['sha256']='0'*64
            (stage/name).unlink();pin=adapter.write(stage/name,producer)
            next(r for r in doc['input_objects'] if r['path']==name).update(pin)
            archive=json.loads((stage/'ARCHIVE-RECEIPT.json').read_bytes())
            next(r for r in archive['science_archive']['objects'] if r['name']==name).update(pin)
            (stage/'ARCHIVE-RECEIPT.json').unlink()
            pin=adapter.write(stage/'ARCHIVE-RECEIPT.json',archive)
            next(r for r in doc['input_objects'] if r['path']=='ARCHIVE-RECEIPT.json').update(pin)
            doc['input_body_bytes']=sum(r['bytes'] for r in doc['input_objects'])
            with self.assertRaisesRegex(ValueError,'producer_binds_cached_outputs'):
                adapter.load(stage,doc,synthetic=True)

    def test_real_scope_rejects_synthetic(self):
        with self.assertRaisesRegex(ValueError,'complete_source_bridge'):A.aggregate(*fixture())

    def test_Mac_gate_precedes_input_read(self):
        with patch.object(adapter.sys,'platform','darwin'),patch.object(
                Path,'read_bytes',side_effect=AssertionError('must not read')):
            with self.assertRaisesRegex(ValueError,'cloud_Linux_only'):
                adapter.run('/missing','/missing','',0)


if __name__=='__main__':unittest.main()
