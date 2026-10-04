import hashlib

from aquasol_meta.reproducibility import _included_files, _locked_packages, _sha256


def test_sha256_reads_file_content(tmp_path) -> None:
    path = tmp_path / "example.txt"
    path.write_text("aqueous solubility\n", encoding="utf-8")

    assert _sha256(path) == hashlib.sha256(b"aqueous solubility\n").hexdigest()


def test_locked_packages_ignores_comments_and_parses_versions(tmp_path) -> None:
    (tmp_path / "requirements-lock.txt").write_text(
        "# snapshot\npandas==2.3.3\nscikit-learn==1.9.1\n",
        encoding="utf-8",
    )

    assert _locked_packages(tmp_path) == {
        "pandas": "2.3.3",
        "scikit-learn": "1.9.1",
    }


def test_included_files_excludes_self_referential_reports_and_cache(tmp_path) -> None:
    (tmp_path / "README.md").write_text("read me", encoding="utf-8")
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "ordinary.json").write_text("{}", encoding="utf-8")
    (reports / "reproducibility_manifest.json").write_text("{}", encoding="utf-8")
    cache = tmp_path / "results" / "chemprop_cache"
    cache.mkdir(parents=True)
    (cache / "checkpoint.pt").write_bytes(b"model")

    included = {path.relative_to(tmp_path).as_posix() for path in _included_files(tmp_path)}

    assert included == {"README.md", "reports/ordinary.json"}
