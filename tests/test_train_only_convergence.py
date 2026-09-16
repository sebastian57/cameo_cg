from types import SimpleNamespace

from utils.jax_setup import apply_jax_compat_shims

apply_jax_compat_shims()


def test_train_only_convergence_records_loss_without_validation(capsys):
    from training.trainer import _evaluate_train_only_convergence

    chemtrain = SimpleNamespace(
        _batches_per_epoch={"training": 2},
        train_batch_losses=[1.0, 3.0],
        train_losses=[],
        val_losses=[],
        update_times=[1.25],
        _epoch=0,
        gradient_norm_history=[0.5],
        train_target_losses={"F": [2.0, 4.0]},
        val_target_losses={},
    )

    _evaluate_train_only_convergence(chemtrain)

    assert chemtrain.train_losses == [2.0]
    assert chemtrain.val_losses == []
    output = capsys.readouterr().out
    assert "not evaluated (no held-out split)" in output
    assert "F | train loss: 3.0 | val loss: N.A." in output


def test_train_only_convergence_replaces_chemtrain_task():
    from training.trainer import (
        _evaluate_train_only_convergence,
        _install_train_only_convergence_task,
    )

    class FakeChemtrain:
        def __init__(self):
            self._tasks = {"post_epoch": [self._evaluate_convergence, lambda *_: None]}

        def _evaluate_convergence(self):
            raise AssertionError("Chemtrain validation convergence task was not removed")

        def add_task(self, trigger, fn):
            self._tasks.setdefault(trigger, []).append(fn)

    chemtrain = FakeChemtrain()
    _install_train_only_convergence_task(chemtrain)

    assert all(
        getattr(fn, "__name__", None) != "_evaluate_convergence"
        for fn in chemtrain._tasks["post_epoch"]
    )
    assert _evaluate_train_only_convergence in chemtrain._tasks["post_epoch"]
