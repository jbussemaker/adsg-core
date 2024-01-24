import ctypes
import threading
import multiprocessing.pool

try:
    import gevent
    import gevent.monkey
except ImportError:
    gevent = None

__all__ = ['run_timeout']


def run_timeout(seconds: float, func, *args, **kwargs):

    # Use native thread pool of gevent, as patched threads behave like cooperative greenlets,
    # which block if not explicitly giving control
    if gevent is not None and gevent.monkey.is_module_patched('threading'):
        raise RuntimeError('Time limiter not compatible with monkey-patched gevent threading module!')

    with multiprocessing.pool.ThreadPool(processes=1) as pool:
        thread = pool.apply(lambda: threading.current_thread())

        try:
            return pool.apply_async(func, args, kwargs).get(timeout=seconds)
        except multiprocessing.TimeoutError:
            pass

    if thread.is_alive():
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(thread.ident), ctypes.py_object(KeyboardInterrupt()))
        thread.join()
    raise TimeoutError
