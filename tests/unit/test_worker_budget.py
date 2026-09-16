"""
Unit tests for worker_budget, CNMF worker sizing.

Pure arithmetic only: nothing here starts a process or probes memory.
"""
from worker_budget import GB, chunk_pixels, plan_workers, reserve_bytes


class TestPlanWorkers:

    def test_plenty_of_memory_hits_cap(self):
        assert plan_workers(200 * GB, 256 * GB, 1 * GB, max_workers=19) == 19

    def test_higher_cap_is_allowed_when_memory_allows(self):
        assert plan_workers(200 * GB, 256 * GB, 1 * GB, max_workers=40) == 40

    def test_limited_by_free_memory(self):
        # 34 GB machine, 18 GB free: reserve max(4, 5.1) GB -> 12.9 GB budget
        n = plan_workers(18 * GB, 34 * GB, int(1.2 * GB), max_workers=19)
        assert n == 10

    def test_never_below_one(self):
        assert plan_workers(1 * GB, 34 * GB, 1 * GB, max_workers=19) == 1

    def test_reserve_floor(self):
        assert reserve_bytes(8 * GB) == 4 * GB
        assert reserve_bytes(100 * GB) == 15 * GB


class TestChunkPixels:

    def test_chunk_shrinks_with_longer_movies(self):
        assert chunk_pixels(1028) > chunk_pixels(4000)

    def test_chunk_near_target(self):
        npx = chunk_pixels(1028)
        assert 100 * 1024 ** 2 < npx * 1028 * 32 <= 128 * 1024 ** 2

    def test_minimum_chunk(self):
        assert chunk_pixels(10 ** 7) == 500