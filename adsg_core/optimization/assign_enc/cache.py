import os
import shutil
from appdirs import user_cache_dir


def get_cache_path(sub_path: str = None) -> str:
    cache_folder = user_cache_dir('ADSG')
    os.makedirs(cache_folder, exist_ok=True)
    return os.path.join(cache_folder, sub_path) if sub_path is not None else cache_folder


def reset_caches(sub_path: str = None):
    cache_path = get_cache_path(sub_path=sub_path)
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path, ignore_errors=True)
