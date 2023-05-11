import os
import pytest
from smart_jit import jit  # noqa: F401

try:
    from numba.tests.test_caching import DispatcherCacheUsecasesTest
except ImportError:
    pytest.skip('Missing "DispatcherCacheUsecasesTest"',
                allow_module_level=True)


class TestCache(DispatcherCacheUsecasesTest):
    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, "smart_jit_cache_usecases.py")
    modname = "dispatcher_caching_test_fodder"

    def test_caching(self):
        self.check_pycache(0)
        mod = self.import_module()
        self.check_pycache(0)

        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_pycache(2)  # 1 index, 1 data
        self.assertPreciseEqual(f(2.5, 3), 6.5)
        self.check_pycache(3)  # 1 index, 2 data
        self.check_hits(f, 0, 2)

        f = mod.add_objmode_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_pycache(5)  # 2 index, 3 data
        self.assertPreciseEqual(f(2.5, 3), 6.5)
        self.check_pycache(6)  # 2 index, 4 data
        self.check_hits(f, 0, 2)

        # Check the code runs ok from another process
        self.run_in_separate_process()
