"""Thread safety (many ``asyncio.to_thread`` callers) and a throughput sanity check."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tests.unit.storage.conftest import (
    create_scan,
    make_analysis,
    make_point,
    make_profile,
)

from confocal.models import ScanState
from confocal.storage import SQLiteHDF5Repository

N_WRITERS = 4
POINTS_PER_WRITER = 25


def test_concurrent_appends_interleaved_with_reads(repo: SQLiteHDF5Repository) -> None:
    """Writers share one active scan file; readers hit the same cached handle meanwhile."""
    create_scan(repo, "shared")
    create_scan(repo, "other")
    profiles = {
        point_id: make_profile(np.random.default_rng(point_id), n_coarse=6, n_fine=4)
        for point_id in range(N_WRITERS * POINTS_PER_WRITER)
    }
    start = threading.Barrier(N_WRITERS + 2)
    stop_reading = threading.Event()
    read_errors: list[BaseException] = []

    def write(worker: int) -> None:
        start.wait()
        for k in range(POINTS_PER_WRITER):
            point_id = k * N_WRITERS + worker  # interleaved ids across writers
            repo.append_point("shared", make_point(point_id), profiles[point_id], make_analysis())
            if k % 5 == 0:
                repo.append_point("other", make_point(point_id), None, None)
            repo.record_event("point", f"{worker}:{k}", scan_id="shared")

    def read() -> None:
        start.wait()
        try:
            while not stop_reading.is_set():
                points = repo.get_points("shared")
                ids = [p.point_id for p in points]
                assert ids == sorted(ids)
                for point in points[-3:]:
                    record = repo.get_profile("shared", point.point_id)
                    assert record.raw_counts == profiles[point.point_id].raw_counts.tolist()
                repo.list_scans()
        except BaseException as exc:  # reported in the main thread
            read_errors.append(exc)

    with ThreadPoolExecutor(max_workers=N_WRITERS + 2) as pool:
        readers = [pool.submit(read) for _ in range(2)]
        writers = [pool.submit(write, w) for w in range(N_WRITERS)]
        for future in writers:
            future.result()
        stop_reading.set()
        for future in readers:
            future.result()

    assert read_errors == []
    total = N_WRITERS * POINTS_PER_WRITER
    assert [p.point_id for p in repo.get_points("shared")] == list(range(total))
    assert repo.get_scan("shared").completed_points == total
    assert repo.get_scan("other").completed_points == N_WRITERS * 5
    assert len(repo.list_events(limit=1000)) == total
    for point_id in (0, total // 2, total - 1):
        record = repo.get_profile("shared", point_id)
        assert record.voltage_v == profiles[point_id].voltage_v.tolist()


async def test_asyncio_to_thread_callers(repo: SQLiteHDF5Repository) -> None:
    await asyncio.to_thread(create_scan, repo, "async")
    profiles = [make_profile(np.random.default_rng(i), n_coarse=5, n_fine=3) for i in range(8)]
    await asyncio.gather(
        *(
            asyncio.to_thread(repo.append_point, "async", make_point(i), profile, None)
            for i, profile in enumerate(profiles)
        )
    )
    records = await asyncio.gather(
        *(asyncio.to_thread(repo.get_profile, "async", i) for i in range(len(profiles)))
    )
    for record, profile in zip(records, profiles, strict=True):
        assert record.raw_counts == profile.raw_counts.tolist()
    summary = await asyncio.to_thread(repo.update_scan, "async", state=ScanState.COMPLETE)
    assert summary.completed_points == len(profiles)


def test_thousand_points_with_profiles_is_fast(
    repo: SQLiteHDF5Repository, rng: np.random.Generator
) -> None:
    """1000 points x 62 positions x 4 samples, appended and read back in a few seconds.

    ``durable=False`` (no fsync) so the check measures the storage code, not the
    test machine's disk; a real SD card adds its fsync latency per point.
    """
    create_scan(repo, "big")
    profile = make_profile(rng, n_coarse=50, n_fine=12)
    analysis = make_analysis()
    t0 = time.perf_counter()
    for point_id in range(1000):
        repo.append_point(
            "big", make_point(point_id, n_z_positions=profile.n_positions), profile, analysis
        )
    append_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    points = repo.get_points("big")
    page = repo.get_points("big", since_point_id=900, limit=50)
    record = repo.get_profile("big", 999)
    read_s = time.perf_counter() - t0

    assert len(points) == 1000
    assert [p.point_id for p in page] == list(range(901, 951))
    assert record.raw_counts == profile.raw_counts.tolist()
    assert repo.get_scan("big").completed_points == 1000
    assert append_s < 20.0, f"1000 appends took {append_s:.1f} s"
    assert read_s < 5.0, f"reads took {read_s:.1f} s"
