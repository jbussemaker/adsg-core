import pickle
from cached_property import cached_property
from adsg_core.func_cache import cached_function, clear_func_cache


class MyClass:

    def __init__(self):
        self.func_calls = 0

    @cached_function
    def cached_func(self, a, b, c=3, d=4):
        self.func_calls += 1
        return a+b+c+d

    @cached_property
    def property(self):
        self.func_calls += 1
        return 5


def test_cached_serializable():
    obj = MyClass()

    for _ in range(5):
        assert obj.cached_func(1, 2, c=3, d=4) == 10
        assert obj.func_calls == 1

    for _ in range(5):
        assert obj.cached_func(2, 2, d=5) == 12
        assert obj.func_calls == 2

    for _ in range(5):
        assert obj.property == 5
        assert obj.func_calls == 3

    nc = obj.func_calls
    obj2: MyClass = pickle.loads(pickle.dumps(obj))
    assert obj2.func_calls == nc

    for _ in range(2):
        assert obj2.cached_func(1, 2, c=3, d=4) == 10
        assert obj2.func_calls == nc

        assert obj2.cached_func(2, 2, d=5) == 12
        assert obj2.func_calls == nc

        assert obj2.property == 5
        assert obj2.func_calls == nc

        clear_func_cache(obj2, func=obj2.cached_func)
        assert obj2.cached_func(1, 2, c=3, d=4) == 10
        assert obj2.func_calls == nc+1

        assert obj2.cached_func(2, 2, d=5) == 12
        assert obj2.func_calls == nc+2
        nc += 2

    clear_func_cache(obj2)
    assert obj2.cached_func(1, 2, c=3, d=4) == 10
    assert obj2.func_calls == nc+1

    assert obj2.cached_func(2, 2, d=5) == 12
    assert obj2.func_calls == nc+2

    assert obj2.property == 5
    assert obj2.func_calls == nc+2
