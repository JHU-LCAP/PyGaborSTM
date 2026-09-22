"""Package surface: lazy submodules, exports, version.

Separate from the per-module test files because the behaviour under test
belongs to ``pygaborstm/__init__.py`` itself rather than any one stage.
"""

import json
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
        pytest.importorskip("matplotlib", reason="needs pygaborstm[viz]")
        assert stm.plot.__name__ == "pygaborstm.plot"
        assert stm.analysis.__name__ == "pygaborstm.analysis"

    def test_from_import_works(self):
        pytest.importorskip("matplotlib", reason="needs pygaborstm[viz]")
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


class TestBuildInfo:
    def test_always_returns_a_dict_with_the_expected_keys(self):
        info = stm.build_info()
        assert set(info) == {"version", "source", "commit", "url"}
        assert info["version"] == stm.__version__

    def test_source_is_a_known_value(self):
        assert stm.build_info()["source"] in {
            "git",
            "archive",
            "local",
            "editable",
            "index",
            "unknown",
        }

    def test_reads_the_commit_from_a_git_record(self, monkeypatch):
        sha = "a" * 40
        record = json.dumps(
            {
                "url": "https://github.com/JHU-LCAP/PyGaborSTM",
                "vcs_info": {
                    "vcs": "git",
                    "requested_revision": "main",
                    "commit_id": sha,
                },
            }
        )
        _patch_direct_url(monkeypatch, record)

        info = stm.build_info()
        assert info["source"] == "git"
        assert info["commit"] == sha

    def test_archive_install_has_no_commit(self, monkeypatch):
        _patch_direct_url(
            monkeypatch,
            json.dumps(
                {"url": "file:///tmp/x.whl", "archive_info": {"hash": "sha256=ab"}}
            ),
        )
        info = stm.build_info()
        assert info["source"] == "archive"
        assert info["commit"] is None

    def test_editable_install_is_reported_as_such(self, monkeypatch):
        _patch_direct_url(
            monkeypatch,
            json.dumps({"url": "file:///repo", "dir_info": {"editable": True}}),
        )
        assert stm.build_info()["source"] == "editable"

    def test_malformed_record_does_not_raise(self, monkeypatch):
        # Diagnostics must degrade, not explode.
        _patch_direct_url(monkeypatch, "{not json")
        info = stm.build_info()
        assert info["source"] == "unknown"
        assert info["commit"] is None

    def test_missing_record_is_treated_as_an_index_install(self, monkeypatch):
        _patch_direct_url(monkeypatch, None)
        info = stm.build_info()
        assert info["source"] == "index"
        assert info["commit"] is None

    @pytest.mark.parametrize(
        "payload",
        [
            "null",
            "[]",
            '{"vcs_info": null}',
            '{"dir_info": []}',
            "",
            '{"url": [1], "archive_info": {}}',
        ],
    )
    def test_structurally_broken_records_never_raise(self, monkeypatch, payload):
        # Valid JSON of the wrong shape is not a syntax error, so catching
        # ValueError alone let these through as AttributeError.
        _patch_direct_url(monkeypatch, payload)
        info = stm.build_info()
        assert set(info) == {"version", "source", "commit", "url"}
        assert info["commit"] is None

    def test_a_non_git_vcs_is_not_reported_as_git(self, monkeypatch):
        _patch_direct_url(
            monkeypatch,
            json.dumps({"url": "x", "vcs_info": {"vcs": "hg", "commit_id": "abc"}}),
        )
        info = stm.build_info()
        assert info["source"] == "vcs"
        assert info["commit"] is None

    def test_a_non_string_commit_is_discarded(self, monkeypatch):
        _patch_direct_url(
            monkeypatch,
            json.dumps({"url": "x", "vcs_info": {"vcs": "git", "commit_id": 123}}),
        )
        assert stm.build_info()["commit"] is None


def _patch_direct_url(monkeypatch, payload):
    """Make importlib.metadata return ``payload`` for direct_url.json.

    The fake asserts the distribution name it was asked for. Without that, a
    lookup of the wrong package still satisfies every test below.
    """
    import importlib.metadata as md

    class FakeDist:
        @staticmethod
        def read_text(name):
            assert name == "direct_url.json"
            return payload

        @staticmethod
        def locate_file(name):
            # Claim to be the module that actually got imported, so the
            # identity check passes and the parsing branches are reached.
            assert name == "pygaborstm/__init__.py"
            return stm.__file__

    def fake_distribution(name):
        assert name == "pygaborstm", f"looked up the wrong distribution: {name!r}"
        return FakeDist()

    monkeypatch.setattr(md, "distribution", fake_distribution)


class TestExports:
    def test_all_names_are_reachable(self):
        # __all__ deliberately excludes plot/analysis, so this stays true on a
        # bare install.
        for name in stm.__all__:
            assert getattr(stm, name) is not None

    def test_version_matches_the_installed_distribution(self):
        # Asserting only "is a non-empty string" passes on the
        # "0.0.0.dev0" fallback, so it cannot detect broken metadata wiring.
        from importlib.metadata import version

        assert stm.__version__ == version("pygaborstm")
        assert stm.__version__ != "0.0.0.dev0"
