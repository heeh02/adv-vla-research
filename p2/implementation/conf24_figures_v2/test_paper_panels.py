"""Explicit synthetic JSON fixtures and offline paper-panel/component checks."""
from pathlib import Path
import argparse
import copy
import json
import tempfile
import unittest
from unittest.mock import patch

import adapter
import audit
import paper_panels as P
import renderer
from synthetic_fixture import fixture as pvp_fixture


def fake_values():
    """No real inputs or filesystem search: values are constructed here."""
    f1 = {}
    for mi, mode in enumerate(P.MODES):
        for pi, parent in enumerate(('C', 'T')):
            f1[parent+'_'+mode] = [
                [(.20+.18*fi+.07*di+.12*pi)*(1.+.4*mi) for di in range(7)]
                for fi in range(9)]
        f1['DEV_'+mode] = {
            f: {'ratios': [(.02+.025*fi+.014*di)*(1.+.3*mi) for di in range(7)],
                'defined_dimensions': 7}
            for fi, f in enumerate(P.FAMILIES)}
    f1['DEV_lambda1']['current4']['ratios'][0] = 0.
    f1['DEV_lambda1']['current4']['ratios'][1] = None
    f1['DEV_lambda1']['current4']['defined_dimensions'] = 6
    f1['C_edf_match'][P.FAMILIES.index('current_full')][0] = 0.
    f1['T_edf_match'][P.FAMILIES.index('headpos_signed')][1] = None
    f3 = {w: [[.25+.18*r+.09*d+.13*wi for d in range(7)] for r in range(8)]
          for wi, w in enumerate(('First 5', 'Full 10'))}
    f3['First 5'][0][0] = 0.
    f3['Full 10'][1][1] = None
    return {'schema': 'signed-figures-values-v1', 'synthetic': True,
            'dimensions': ['synthetic_dim_'+str(i) for i in range(7)],
            'model_names': ['synthetic_model_'+str(i) for i in range(37)],
            'main': {'F1-CAPACITY-TRANSFER': f1, 'F3-ROOT-COUNTEREXAMPLES': f3}}


def write_fixture(stage, *, values=None, mutate_producer=None):
    """Create exactly three synthetic cached JSON objects and pins."""
    stage = Path(stage)
    stage.mkdir(parents=True, exist_ok=False)
    values = fake_values() if values is None else values
    qa = {'synthetic': True, 'passed': True, 'all37_predictors': True,
          'CONF_KT_rows': 96, 'DEV_rows': 48, 'fits_retrained': 0}
    refs = []
    for name, value in zip(P.EXTRA_INPUTS[:2], (values, qa)):
        raw = P.encoded(value)
        (stage/name).write_bytes(raw)
        refs.append({'path': name, **P.identity(raw)})
    producer = {'synthetic': True, 'analysis_complete': True, 'render_complete': True,
                'input_qualified': True, 'new_model_calls': 0, 'new_env_steps': 0,
                'new_fits': 0, 'training_updates': 0, 'JPFS_new_bytes': 0,
                'outputs': copy.deepcopy(refs)}
    if mutate_producer is not None:
        mutate_producer(producer)
    raw = P.encoded(producer)
    (stage/P.EXTRA_INPUTS[2]).write_bytes(raw)
    refs.append({'path': P.EXTRA_INPUTS[2], **P.identity(raw)})
    return {'input_objects': refs, 'input_body_bytes': sum(r['bytes'] for r in refs)}


class PaperPanelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.stage = self.base/'synthetic-cache'
        self.manifest = write_fixture(self.stage)

    def tearDown(self):
        self.temp.cleanup()

    def test_three_JSON_qualified_and_all_cells_preserved(self):
        values, refs = P.load(self.stage, synthetic=True, manifest=self.manifest)
        self.assertEqual(set(refs), set(P.EXTRA_INPUTS))
        with P.plt.rc_context(P.STYLE):
            fig, data = P.transfer(values, synthetic=True)
        P.plt.close(fig)
        self.assertEqual(len(data['scatter_points']), 126)
        self.assertEqual(data['root_cells'], 112)
        self.assertEqual(data['training_display']['points'], 42)
        self.assertEqual(data['evaluation_display']['points'], 84)
        self.assertEqual(data['matrix_zero_count'], 1)
        self.assertEqual(data['root_undefined_count'], 1)
        self.assertIsNone(data['root_matrix'][1][8])
        self.assertEqual(data['root_matrix'][0][0], 0.)
        self.assertEqual(sum(r['zero_marker'] for r in data['scatter_points']), 2)
        self.assertEqual(sum(r['undefined_marker'] for r in data['scatter_points']), 2)
        self.assertTrue(all(r['relative_MSE'] == 0 for r in data['scatter_points'] if r['zero_marker']))
        self.assertTrue(all(r['relative_MSE'] is None for r in data['scatter_points'] if r['undefined_marker']))

    def test_wide_synthetic_log_range_keeps_axis_title_inside(self):
        values = fake_values()
        f1 = values['main']['F1-CAPACITY-TRANSFER']
        for mode in P.MODES:
            for family in P.FAMILIES:
                f1['DEV_'+mode][family]['ratios'] = [10.**(-12+2*i) for i in range(7)]
        with P.plt.rc_context(P.STYLE):
            fig, data = P.transfer(values, synthetic=True)
            layout = P.save(fig, self.base, 'SYNTHETIC-WIDE')
        self.assertEqual(layout['text_bbox_check'], 'PASS')
        self.assertGreaterEqual(layout['minimum_font_points'], 7.5)
        self.assertEqual(len(data['scatter_points']), 126)

    def test_bad_input_SHA_rejected(self):
        bad = copy.deepcopy(self.manifest)
        bad['input_objects'][0]['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'cached_panel_input_SHA'):
            P.load(self.stage, synthetic=True, manifest=bad)

    def test_wrong_producer_binding_rejected_after_manifest_pass(self):
        stage = self.base/'synthetic-wrong-binding'
        manifest = write_fixture(stage, mutate_producer=lambda p: p['outputs'][0].update(sha256='0'*64))
        with self.assertRaisesRegex(ValueError, 'producer_binds_exact_scalar_cache'):
            P.load(stage, synthetic=True, manifest=manifest)

    def test_synthetic_scope_mismatch_rejected(self):
        v = fake_values(); v['synthetic'] = False
        stage = self.base/'wrong-scope'; manifest = write_fixture(stage, values=v)
        with self.assertRaisesRegex(ValueError, 'same_declared_cache_scope'):
            P.load(stage, synthetic=True, manifest=manifest)

    def test_actual_Mac_denied_before_read(self):
        with patch.object(P.sys, 'platform', 'darwin'), patch.object(
                Path, 'read_bytes', side_effect=AssertionError('must not read')):
            with self.assertRaisesRegex(ValueError, 'actual_panel_data_only_on_cloud_Linux'):
                P.load(self.stage, synthetic=False, manifest=self.manifest)

    def test_actual_shape_and_negative_values_rejected(self):
        for bad in ('shape', 'negative'):
            with self.subTest(bad=bad):
                values = fake_values()
                target = values['main']['F1-CAPACITY-TRANSFER']['C_edf_match']
                if bad == 'shape':
                    target.pop()
                else:
                    target[0][0] = -1.
                stage = self.base/('wrong-'+bad)
                manifest = write_fixture(stage, values=values)
                with self.assertRaisesRegex(ValueError, 'complete_fixed_population_cells'):
                    P.load(stage, synthetic=True, manifest=manifest)

    def test_render_exact_eight_output_files(self):
        out = self.base/'panels'; out.mkdir()
        result = P.render(self.stage, out, lambda: None, synthetic=True, manifest=self.manifest)
        self.assertEqual(set(result['outputs']), set(adapter.PAPER_OUTPUT_NAMES))
        self.assertEqual({p.name for p in out.iterdir()}, set(adapter.PAPER_OUTPUT_NAMES))
        self.assertEqual(result['numerical_QA']['scatter_points'], 126)
        self.assertEqual(result['numerical_QA']['all_eight_roots_both_windows_coordinates'], 112)
        self.assertTrue(all(r['minimum_font_points'] >= 7.5 for r in result['layouts']))


def integrated_synthetic(out):
    """Components only: never bypass the real adapter's cloud-only gate."""
    root = Path(out); root.mkdir(parents=True, exist_ok=False)
    stage = root/'synthetic-inputs'
    manifest = write_fixture(stage)
    result_dir = root/'synthetic-outputs'; result_dir.mkdir()
    tables, producer = pvp_fixture()
    data = audit.verify(tables, producer, synthetic=True)
    plotted = renderer.render(data, result_dir)
    panels = P.render(stage, result_dir, lambda: None, synthetic=True, manifest=manifest)
    adapter.write(result_dir/'FULL-CACHED-SUPPORT.json.gz',
                  {'synthetic': True, 'tables': tables}, compressed=True)
    adapter.write(result_dir/'FIGURE-VALUES.json', {'synthetic': True, **plotted['main_values']})
    adapter.write(result_dir/'CAPTIONS.json', plotted['captions'])
    adapter.write(result_dir/'QA.json', data['qa'])
    adapter.write(result_dir/'RENDER-MANIFEST.json', {
        'synthetic': True, 'core_layouts': plotted['core_layouts'],
        'atlas_pages': plotted['atlas_pages'], 'paper_layouts': panels['layouts'],
        'outputs': [{'path': p.name, **adapter.ident(p)} for p in sorted(result_dir.iterdir())]})
    adapter.write(result_dir/adapter.RESULT, {
        'schema': 'paper2-conf24-figures-result-v1', 'synthetic': True,
        'status': 'SYNTHETIC_COMPONENTS_ONLY', 'analysis_complete': False,
        'actual_cloud_acceptance': False, 'main_figures': 3, 'atlas_pages': 9,
        'outputs': [{'path': p.name, **adapter.ident(p)} for p in sorted(result_dir.iterdir())]})
    actual = {p.name for p in result_dir.iterdir()}
    assert actual == set(adapter.OUTPUT_NAMES)
    summary = {'synthetic': True, 'scientific_result': False,
               'actual_adapter_executed': False, 'output_roster': sorted(actual),
               'outputs': len(actual), 'scatter_points': panels['numerical_QA']['scatter_points'],
               'root_cells': panels['numerical_QA']['all_eight_roots_both_windows_coordinates'],
               'command_cells': len(data['command_cells']),
               'total_bytes': sum(p.stat().st_size for p in result_dir.iterdir()),
               'largest_file_bytes': max(p.stat().st_size for p in result_dir.iterdir())}
    adapter.write(root/'INTEGRATED-SYNTHETIC-QA.json', summary)
    print(json.dumps(summary))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render-integrated-synthetic')
    args, rest = parser.parse_known_args()
    if args.render_integrated_synthetic:
        integrated_synthetic(args.render_integrated_synthetic)
    else:
        unittest.main(argv=[__file__, *rest])
