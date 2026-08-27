"""Tests for concurrency."""
import threading
import time

from aion_knowledge.pipeline.parser.concurrency import parser_worker_limit


class TestConcurrency:
    def test_worker_limit_blocks_concurrent_access(self):
        results = []
        def worker():
            with parser_worker_limit("test", 1):
                results.append(1)
                time.sleep(0.05)
        threads = [threading.Thread(target=worker) for _ in range(3)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - t0
        assert len(results) == 3
        assert elapsed >= 0.1

    def test_worker_limit_disabled(self):
        with parser_worker_limit("test_disabled", 0):
            pass
