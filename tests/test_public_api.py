"""Package surface: lazy submodules, exports, version.

Separate from the per-module test files because the behaviour under test
belongs to ``pygaborstm/__init__.py`` itself rather than any one stage.
"""

import subprocess
import sys

import pytest

import pygaborstm as stm


class TestLazySubmodules:
    def test_importing_the_package_does_not_import_matplotlib(self):
        # Must run in a fresh interpreter: any earlier test that touched
        # stm.plot would leave matplotlib in this process's sys.modules.
        code = (
            "import sys, pygaborstm; "
            "print('matplotlib' in sys.modules or 'seaborn' in sys.modules)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "False"

    def test_attribute_access_loads_the_submodule(self):
        assert stm.plot.__name__ == "pygaborstm.plot"
        assert stm.analysis.__name__ == "pygaborstm.analysis"

    def test_from_import_works(self):
        from pygaborstm import plot

        assert plot.plt_rsf is not None

    def test_lazy_names_appear_in_dir(self):
        listed = dir(stm)
        assert "plot" in listed
        assert "analysis" in listed

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="no attribute 'nope'"):
            stm.nope  # noqa: B018 - the bare access is what raises

    def test_missing_extra_names_the_install_command(self):
        # Simulate matplotlib being absent, as on a bare `pip install`.
        code = (
            "import sys; sys.modules['matplotlib'] = None; "
            "import pygaborstm\n"
            "try:\n"
            "    pygaborstm.plot\n"
            "except ImportError as e:\n"
            "    print('OK' if \"pygaborstm[viz]\" in str(e) else 'NOHINT')\n"
            "else:\n"
            "    print('NOERROR')\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "OK", out.stderr


class TestExports:
    def test_all_names_are_reachable(self):
        for name in stm.__all__:
            assert getattr(stm, name) is not None

    def test_version_is_set(self):
        assert isinstance(stm.__version__, str)
        assert stm.__version__
