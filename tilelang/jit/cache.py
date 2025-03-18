import dataclasses
import functools
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
import hashlib


def _get_workspace_dir_name() -> Path:
    return Path(os.getenv("TL_WORKSPACE_DIR", Path.home() / ".cache" / "tilelang"))


@dataclasses.dataclass
class FileGroup:
    source: str
    library: str

class CacheManager(ABC):

    @abstractmethod
    def get_file_group(self, ext, code, always_new=False) -> FileGroup:
        pass

    @abstractmethod
    def clear_all(self):
        pass


class FileCacheManager(CacheManager):

    def __init__(self):
        self.cache_dir = _get_workspace_dir_name()
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_file_group(self, ext, code, always_new=False) -> FileGroup:
        if always_new:
            with tempfile.NamedTemporaryFile(mode="w", suffix=f".{ext}", delete=False) as src:
                src.write(code)
                src.flush()
                return FileGroup(source=src.name, library=src.name.replace(f".{ext}", ".so"))
        code_hash = hashlib.md5(code.encode("utf-8")).hexdigest()
        cached_dir = self.cache_dir / str(code_hash)
        if not cached_dir.exists():
            cached_dir.mkdir(parents=True)
        ext = ext.strip(".")
        source_file = cached_dir / f"src_gen.{ext}"
        lib_file = cached_dir / "lib_gen.so"
        if not source_file.exists():
            with open(source_file, "w") as f:
                f.write(code)
                f.flush()
        return FileGroup(source=str(source_file), library=str(lib_file))

    def clear_all(self):
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir()


class RemoteCacheManager(CacheManager):
    def __init__(self):
        pass

    def get_file_group(self, ext, code, always_new=False) -> FileGroup:
        pass

    def clear_all(self):
        pass


@functools.cache
def get_cache_manager() -> CacheManager:
    return FileCacheManager()
