import pickle
import hashlib
import functools


def cached_function(func):
    """
    Function decorator for caching function calls using the class dict, so it can be serialized (pickled).

    Inspired by:
    - https://stackoverflow.com/a/15587418
    - cached_property
    """
    name = func.__name__

    def wrapper(obj, *args, **kwargs):
        if obj is None:
            raise RuntimeError('cached_serializable should be applied to a regular class function,'
                               'not a @staticmethod or @classmethod!')
        cache = obj.__dict__

        # Python-session persistent hashing, because the hash() function is randomized between Python instances
        # This is slower than normal hashing, so that's why we also apply lru_cache below
        cache_args = b'||'.join((b'__'.join(pickle.dumps(v) for v in args),
                                 b'__'.join(str(k).encode('utf-8')+b'='+pickle.dumps(v) for k, v in kwargs.items())))
        cache_key = '_cache_'+name+'_'+hashlib.md5(cache_args).hexdigest()[:8]

        if cache_key in cache:
            return cache[cache_key]
        cache[cache_key] = value = func(obj, *args, **kwargs)
        return value

    wrapper.__name__ = name
    return functools.lru_cache()(wrapper)


def clear_func_cache(obj, func=None):
    key_start = '_cache_'
    if func is not None:
        func_name = func if isinstance(func, str) else func.__name__
        key_start += func_name+'_'

    # Clear persistent cache
    cleared_funcs = set()
    for key in list(obj.__dict__.keys()):
        if key.startswith(key_start):
            cleared_funcs.add('_'.join(key.split('_')[2:-1]))
            del obj.__dict__[key]

    # Clear lru_cache
    for cleared_func_name in cleared_funcs:
        func = getattr(obj, cleared_func_name)
        if hasattr(func, 'cache_clear'):
            func.cache_clear()
