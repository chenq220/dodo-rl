"""pytest 运行产物清理：只删除本轮新建的 logs/artifacts 路径。"""
from __future__ import annotations

from pathlib import Path

from conftest import _cleanup_new_paths, _snapshot_tree


def test_cleanup_removes_only_new_artifact_paths(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    keep_dir = root / "models"
    keep_dir.mkdir(parents=True)
    keep_file = keep_dir / "keep.pth"
    keep_file.write_text("old", encoding="utf-8")
    before = _snapshot_tree(root)

    new_dir = root / "results"
    new_dir.mkdir()
    (new_dir / "curve.png").write_text("new", encoding="utf-8")
    (keep_dir / "new.pth").write_text("new", encoding="utf-8")

    _cleanup_new_paths(root, existed_before=True, before=before)

    assert keep_file.read_text(encoding="utf-8") == "old"
    assert not (keep_dir / "new.pth").exists()
    assert not new_dir.exists()


def test_cleanup_removes_entire_tree_when_absent_before(tmp_path: Path) -> None:
    root = tmp_path / "logs"
    root.mkdir()
    (root / "rl.log").write_text("x", encoding="utf-8")

    _cleanup_new_paths(root, existed_before=False, before=set())

    assert not root.exists()
