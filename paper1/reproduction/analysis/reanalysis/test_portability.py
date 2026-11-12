"""Package-location regressions; these checks add no scientific samples."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import run_e as m


class PortabilityTests(unittest.TestCase):
    def test_root_is_adjacent_package(self):
        self.assertEqual(m.BASE, Path(m.__file__).resolve().parents[1])

    def test_s4_dependency_is_inside_package(self):
        self.assertEqual(m.input_path('statistics/internal_link_pure.py'),
                         m.BASE / 'statistics/internal_link_pure.py')

    def test_missing_dependency_has_explicit_package_path(self):
        with tempfile.TemporaryDirectory(prefix='missing-e-dependency-') as temporary:
            with mock.patch.object(m, 'BASE', Path(temporary)):
                with self.assertRaisesRegex(
                        FileNotFoundError,
                        'Missing packaged reanalysis dependency: statistics/internal_link_pure.py'):
                    m.s4_module()

    def test_relocated_module_ignores_working_directory(self):
        with tempfile.TemporaryDirectory(prefix='relocated-e-package-') as temporary:
            root = Path(temporary) / 'anonymous-package'
            (root / 'reanalysis').mkdir(parents=True)
            (root / 'statistics').mkdir()
            shutil.copyfile(m.__file__, root / 'reanalysis/run_e.py')
            shutil.copyfile(m.input_path('statistics/internal_link_pure.py'),
                            root / 'statistics/internal_link_pure.py')
            code = (
                'import sys; from pathlib import Path; '
                'sys.path.insert(0, sys.argv[1]); import run_e; '
                'assert run_e.BASE == Path(sys.argv[1]).resolve().parent; '
                'assert run_e.s4_module().weighted_auc('
                '[dict(row_id="a", Y_cell="SF", x=2), '
                'dict(row_id="b", Y_cell="SS", x=2)], "x") == .5'
            )
            result = subprocess.run(
                [sys.executable, '-B', '-c', code, str(root / 'reanalysis')],
                cwd=temporary, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
