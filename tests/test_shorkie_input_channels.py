"""Checkpoint input format and regression checks for both Shorkie variants."""

# ruff: noqa: E402 -- optional model dependencies must be checked before imports.
import copy
import json
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")

from yeastbench.models.shorkie.nn import (
    ShorkieModule,
    _tf_conv_dna_in_channels,
    _with_conv_dna_in_channels,
)
from yeastbench.config import load_config
from yeastbench.data.manifest import artifact_by_id
from yeastbench.data.lock import read_lock, locked_files
from yeastbench.adapters.shorkie_hong import _default_targets_path

FIXTURES = Path(__file__).parent / "fixtures" / "shorkie"


@pytest.mark.parametrize("channels", [4, 170])
@pytest.mark.parametrize(
    "rooted,nested", [(True, True), (True, False), (False, True), (False, False)]
)
def test_checkpoint_input_width(tmp_path, channels, rooted, nested):
    path = tmp_path / "weights.h5"
    with h5py.File(path, "w") as f:
        root = f.create_group("model_weights") if rooted else f
        group = root.create_group("conv1d")
        if nested:
            group = group.create_group("conv1d")
        group.create_dataset("kernel:0", shape=(11, channels, 96), dtype="f4")
    assert _tf_conv_dna_in_channels(path) == channels


@pytest.mark.parametrize("shape", [(11, 5, 96), (11, 166, 96), (4, 96)])
def test_rejects_unknown_width_before_build(tmp_path, shape):
    path = tmp_path / "weights.h5"
    with h5py.File(path, "w") as f:
        f.create_group("conv1d").create_dataset("kernel:0", shape=shape, dtype="f4")
    with pytest.raises(ValueError, match=r"4, 170"):
        ShorkieModule.from_tf_checkpoint({}, path)


@pytest.mark.parametrize("channels", [4, 170])
def test_forward_uses_checkpoint_input_format(channels):
    config = json.loads((FIXTURES / "params.json").read_text())["model"]
    original = copy.deepcopy(config)
    configured = _with_conv_dna_in_channels(config, channels)
    model = ShorkieModule(configured).eval()
    assert config == original
    assert model.backbone.conv_dna.conv.in_channels == channels
    assert model._expects_species_encoding == (channels == 170)
    x = torch.zeros(1, 4, 16384)
    x[:, 2, :] = 1
    captured = []

    class StopAfterInput(Exception):
        pass

    def capture(module, args):
        captured.append(args[0])
        raise StopAfterInput

    handle = model.backbone.conv_dna.register_forward_pre_hook(capture)
    try:
        with pytest.raises(StopAfterInput):
            model(x)
    finally:
        handle.remove()
    got = captured[0]
    assert got.shape == (1, channels, 16384)
    torch.testing.assert_close(got[:, :4], x, rtol=0, atol=0)
    if channels == 170:
        assert torch.all(got[:, 114] == 1)
        assert torch.count_nonzero(got[:, 4:]) == 16384
    else:
        assert got is x


def test_default_config_still_builds_species_model():
    config = json.loads((FIXTURES / "params.json").read_text())["model"]
    assert ShorkieModule(config).backbone.conv_dna.conv.in_channels == 170


def test_targets_prefer_adjacent_then_fall_back(tmp_path):
    params = tmp_path / "params.json"
    fallback = _default_targets_path(params)
    assert fallback.as_posix().endswith("/data/models/shorkie/targets.txt")
    adjacent = tmp_path / "targets.txt"
    adjacent.touch()
    assert _default_targets_path(params) == adjacent


def test_random_init_artifact_complete_and_independent():
    artifact = artifact_by_id("shorkie_random_init")
    files = {f.path for f in locked_files(read_lock(), artifact.id)}
    assert files == set(artifact.include)
    assert len(files) == 10
    assert not artifact.requires
    assert (
        artifact.mirrors[0].remote_for("targets.txt").endswith("/shorkie/targets.txt")
    )


@pytest.mark.parametrize("config_name", ["default", "brooks", "meneu"])
def test_random_init_config_matches_pretrained(config_name):
    cfg = load_config(f"configs/{config_name}.yaml")

    def configs(model):
        return {
            t: r.model_config for r in cfg.runs if r.model == model for t in r.tasks
        }

    pretrained, scratch = configs("shorkie"), configs("shorkie_random_init")
    assert pretrained.keys() == scratch.keys()
    if config_name == "default":
        assert len(scratch) == 11
    for task in scratch:
        expected = json.loads(
            json.dumps(pretrained[task]).replace(
                "data/models/shorkie/", "data/models/shorkie_random_init/"
            )
        )
        assert scratch[task] == expected


def test_pretrained_cpu_regression():
    checkpoint = Path("data/models/shorkie/checkpoints/f0.h5")
    if not checkpoint.exists():
        pytest.skip("published pretrained fold 0 is not installed")
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            config = json.loads((FIXTURES / "params.json").read_text())["model"]
            model = ShorkieModule.from_tf_checkpoint(config, checkpoint).eval()
            x = (
                torch.nn.functional.one_hot(torch.randint(4, (1, 16384)), 4)
                .permute(0, 2, 1)
                .float()
            )
            from yeastbench.adapters._shorkie_constants import (
                SHORKIE_T0_RNA_SEQ_TRACK_IDS,
            )

            with torch.inference_mode():
                out = model(x).numpy()[:, :, SHORKIE_T0_RNA_SEQ_TRACK_IDS].mean(axis=2)
            np.testing.assert_array_equal(
                out, np.load(FIXTURES / "pretrained_f0_t0_mean.npy")
            )
    finally:
        torch.set_num_threads(threads)
