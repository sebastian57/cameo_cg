from scripts.run_md_parallel import _build_waves


def test_build_waves_uses_processes_per_gpu():
    waves = _build_waves(n_replicas=100, n_gpus=4, procs_per_gpu=4)

    assert len(waves) == 7
    assert waves[0] == list(range(16))
    assert waves[-1] == list(range(96, 100))
