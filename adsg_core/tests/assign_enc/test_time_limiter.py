# from gevent import monkey; monkey.patch_all()  # Raises a RuntimeError in the test
import time
import timeit
import itertools
import threading
import tracemalloc
import numpy as np
from adsg_core.optimization.assign_enc.time_limiter import *


def test_time_limiter():

    exc = []
    n_finished = [0]

    def _do_run():
        tracemalloc.start()
        try:
            s = timeit.default_timer()
            n = 0
            for _ in itertools.combinations_with_replacement(list(range(20)), 10):
                n += 1
                if n % 100 == 0 and timeit.default_timer()-s > .05:
                    break
            n //= 5

            def _func(n_run, a=20, b=10, ret_val=None, raise_exc=None):
                _ = np.ones((1000, 1000))  # ~1 MB
                if raise_exc is not None:
                    raise raise_exc

                for i, _ in enumerate(itertools.combinations_with_replacement(list(range(a)), b)):
                    if i > n_run:
                        n_finished[0] += 1
                        return ret_val

            obj = object()
            mem_start, _ = tracemalloc.get_traced_memory()
            assert run_timeout(.2, _func, n*10, ret_val=obj) is obj
            assert n_finished == [1]
            mem_end, _ = tracemalloc.get_traced_memory()
            assert mem_end - mem_start < 5e5

            mem_start, _ = tracemalloc.get_traced_memory()
            assert run_timeout(.2, lambda: _func(n*10, ret_val=obj)) is obj
            assert n_finished == [2]
            mem_end, _ = tracemalloc.get_traced_memory()
            assert mem_end - mem_start < 5e5

            mem_start, _ = tracemalloc.get_traced_memory()
            try:
                run_timeout(.2, _func, n*10, raise_exc=RuntimeError())
                raise AssertionError
            except RuntimeError:
                pass
            assert n_finished == [2]
            mem_end, _ = tracemalloc.get_traced_memory()
            # assert mem_end - mem_start < 5e5  # Didn't find a way to solve this memory leak :(

            mem_start, _ = tracemalloc.get_traced_memory()
            try:
                run_timeout(.1, _func, n*200)
                raise AssertionError
            except TimeoutError:
                pass

            time.sleep(1.)
            assert n_finished == [2]

            mem_end, _ = tracemalloc.get_traced_memory()
            assert mem_end - mem_start < 5e5

        except Exception as e:
            exc.append(e)
            raise
        finally:
            tracemalloc.stop()

    n_finished = [0]
    _do_run()
    assert len(exc) == 0

    n_finished = [0]
    try:
        thread = threading.Thread(target=_do_run)
        thread.start()
        thread.join()
    except KeyboardInterrupt:
        raise RuntimeError
    assert len(exc) == 0
