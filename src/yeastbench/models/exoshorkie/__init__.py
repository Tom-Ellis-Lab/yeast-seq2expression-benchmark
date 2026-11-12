"""Distilled ExoShorkie models and single-checkpoint inference."""

from yeastbench.models.exoshorkie.checkpoints import CHECKPOINTS, CheckpointInfo

__all__ = ["CHECKPOINTS", "CheckpointInfo", "ExoShorkie", "ExoShorkieModule", "ExoShorkieEnsemble"]


def __getattr__(name):
    # Registry/data discovery must work without the optional torch/h5py extra.
    if name == "ExoShorkieEnsemble":
        from yeastbench.models.exoshorkie.ensemble import ExoShorkieEnsemble
        return ExoShorkieEnsemble
    if name == "ExoShorkie":
        from yeastbench.models.exoshorkie.wrapper import ExoShorkie
        return ExoShorkie
    if name == "ExoShorkieModule":
        from yeastbench.models.exoshorkie.nn import ExoShorkieModule
        return ExoShorkieModule
    raise AttributeError(name)
