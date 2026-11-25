from types import SimpleNamespace

from utils.file_locker import file_lock_decorator


class Writer:
    @file_lock_decorator("demo.lock")
    def write(self, state_raw, json_file):
        with open(json_file, "w", encoding="utf-8") as fp:
            fp.write(str(state_raw))
        return json_file


def test_file_lock_decorator_writes_and_releases(tmp_path):
    path = tmp_path / "out.txt"
    result = Writer().write("hello", str(path))
    assert path.read_text(encoding="utf-8") == "hello"
    assert result == str(path)
    assert not (tmp_path / "demo.lock").exists()
