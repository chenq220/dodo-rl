"""共享任务日志的文件锁与分片索引；不依赖 logger，避免初始化环。"""
from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock


class SharedLogFiles:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.index_path = self.path.with_name(f".{self.path.name}.parts.json")
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")

    @contextmanager
    def locked(self):
        # 锁文件不随轮转更名；所有进程通过同一把锁协调写入、轮转和快照。
        with FileLock(str(self.lock_path)).acquire(poll_interval=0.01):
            yield self

    def _parts(self) -> list[str]:
        if not self.index_path.exists():
            return []
        parts = json.loads(self.index_path.read_text(encoding="utf-8"))
        pattern = re.compile(re.escape(self.path.stem) + r"_\d{6,}" + re.escape(self.path.suffix))
        if not isinstance(parts, list) or any(
            not isinstance(name, str) or not pattern.fullmatch(name) for name in parts
        ) or len(set(parts)) != len(parts):
            raise ValueError(f"invalid shared log index: {self.index_path}")
        return parts

    def reserve_part(self) -> Path:
        """持锁调用：先持久化分片名再更名，进程退出也不会丢失已切出的分片记录。"""
        parts = self._parts()
        number = len(parts) + 1
        name = f"{self.path.stem}_{number:06d}{self.path.suffix}"
        destination = self.path.with_name(name)
        if destination.exists():
            raise FileExistsError(f"shared log part already exists: {destination}")
        parts.append(name)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as writer:
                json.dump(parts, writer)
                writer.flush()
                os.fsync(writer.fileno())
            os.replace(temporary, self.index_path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return destination

    def files(self) -> list[Path]:
        """持锁调用：只取索引列出的分片和当前文件，不扫描其他任务的日志。"""
        paths = [self.path.with_name(name) for name in self._parts()] + [self.path]
        # 预留分片后、rename 前退出的进程可能留下一条尚未生成的记录。
        return [path for path in paths if path.is_file()]
