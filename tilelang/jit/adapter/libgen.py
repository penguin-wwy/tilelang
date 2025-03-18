# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Optional
from .utils import is_cuda_target, is_hip_target, is_cpu_target
from tilelang import tvm as tvm
from tilelang.contrib.nvcc import get_target_compute_version, get_cuda_version
from tvm.target import Target
import ctypes
import os
import subprocess
import logging
from tilelang.env import TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR
from tilelang.jit.cache import get_cache_manager

logger = logging.getLogger(__name__)


class LibraryGenerator(object):
    srcpath: Optional[str] = None
    libpath: Optional[str] = None
    lib_code: Optional[str] = None

    def __init__(self, target: Target):
        self.target = target

    def update_lib_code(self, lib_code: str):
        self.lib_code = lib_code

    # Assume currently we only support CUDA compilation
    def load_lib(self):
        return ctypes.CDLL(self.libpath)

    def compile_lib(self, timeout: float = None, with_tl: bool = True, disable_cache: bool = False):
        target = self.target
        if is_cuda_target(target):
            compute_version = "".join(get_target_compute_version(target).split("."))
            if compute_version == "90":
                compute_version = "90a"

            command = [
                "nvcc",
                "-std=c++17",
                "-w",  # Disable all warning messages
                "-Xcudafe",
                "--diag_suppress=177",
                "--compiler-options",
                "'-fPIC'",
                "-lineinfo",
                "--shared",
                "-lcuda",
                "-gencode",
                f"arch=compute_{compute_version},code=sm_{compute_version}",
            ]
            compiler_version = get_cuda_version()
            ext = ".cu"

        elif is_hip_target(target):
            command = [
                "hipcc",
                "-std=c++17",
                "-fPIC",
                "--shared",
            ]
            compiler_version = "hipcc"
            ext = ".cpp"
        elif is_cpu_target(target):
            command = ["g++", "-std=c++17", "-fPIC", "-shared"]
            with_tl = False
            command += [
                "-I" + TILELANG_TEMPLATE_PATH,
            ]
            compiler_version = "g++"
            ext = ".cpp"
        else:
            raise ValueError(f"Unsupported target: {target}")

        if with_tl:
            command += [
                "-I" + TILELANG_TEMPLATE_PATH,
                "-I" + CUTLASS_INCLUDE_DIR,
            ]
            command += ["-diag-suppress=20013"]
        code = f"""/*
 * TileLang Generated Code
 * Target: {target}
 * Compiler: {compiler_version}
 * Command: {' '.join(command)} -o {{LIBRARY_FILE}} {{SOURCE_FILE}}
 */

{self.lib_code}
        """
        src_and_lib = get_cache_manager().get_file_group(ext, code, always_new=disable_cache)
        if not os.path.exists(src_and_lib.library):
            command += ["-o", src_and_lib.library, src_and_lib.source]
            try:
                ret = subprocess.run(command, timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning(f"Compilation Timeout! {command}")
                return None
            if ret.returncode != 0:
                logger.warning(f"Compilation Failed! {command}")
                return None
        self.srcpath = src_and_lib.source
        self.libpath = src_and_lib.library

    def remove_lib(self):
        if self.libpath:
            os.remove(self.libpath)
        self.libpath = None

    def get_source_path(self):
        return self.srcpath

    def get_lib_path(self):
        return self.libpath

    def set_lib_path(self, libpath):
        self.libpath = libpath

    def set_src_path(self, srcpath):
        self.srcpath = srcpath
