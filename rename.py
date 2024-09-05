import os
import shutil
import glob

def rename_hip_to_cpp(cpp_files):
    for entry in cpp_files:
        shutil.copy(entry, os.path.splitext(entry)[0] + ".cpp")

fa_sources = glob.glob(
    "csrc/flash_attn/src/*.hip"
)

rename_hip_to_cpp(fa_sources)