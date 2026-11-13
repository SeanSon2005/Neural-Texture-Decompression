import os
import shutil
import subprocess

import numpy as np
import pytest
import torch

from ntc import tex_model
from ntc.atlas import AtlasLayout, ONE, WORDS
from ntc.export import write_export
from ntc.model import NTCMaterial, compute_taps
from ntc.quant import mlp_forward
from ntc.reference import ReferenceDecoder

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VORTEX = os.path.join(REPO, "vortex")
VX_BUILD = os.environ.get("VX_BUILD", os.path.expanduser("~/ntd-build"))


@pytest.fixture(scope="session")
def tex_dump(tmp_path_factory):
    gxx = shutil.which("g++")
    hdr = os.path.join(VX_BUILD, "sw", "VX_types.h")
    if not gxx or not os.path.exists(hdr):
        pytest.skip("needs g++ and a configured Vortex build (set VX_BUILD)")
    exe = tmp_path_factory.mktemp("bin") / "tex_dump"
    subprocess.run([gxx, "-O2", "-std=c++17",
                    "-I", os.path.join(VORTEX, "sw", "common"),
                    "-I", os.path.join(VORTEX, "third_party"),
                    "-I", os.path.join(VX_BUILD, "sw"),
                    os.path.join(REPO, "ntc", "tools", "tex_dump.cpp"), "-o", str(exe)], check=True)
    return str(exe)


@pytest.mark.parametrize("log_w,log_h", [(5, 5), (6, 4), (1, 1)])
@pytest.mark.parametrize("filt", [tex_model.FILTER_POINT, tex_model.FILTER_BILINEAR])
@pytest.mark.parametrize("wrap", [tex_model.WRAP_CLAMP, tex_model.WRAP_REPEAT])
def test_tex_model_matches_vortex_host_model(tex_dump, log_w, log_h, filt, wrap):
    rng = np.random.default_rng(log_w * 100 + log_h * 10 + filt * 2 + wrap)
    words = rng.integers(0, 1 << 32, (1 << log_h, 1 << log_w), dtype=np.uint64).astype(np.uint32)
    n = 4000
    u = rng.integers(-ONE // 4, ONE + ONE // 4, n)  # off-texture too, to cover the wrap
    v = rng.integers(-ONE // 4, ONE + ONE // 4, n)
    payload = (np.array([log_w, log_h, filt, wrap, n], dtype="<u4").tobytes()
               + words.astype("<u4").tobytes()
               + np.stack([u, v], 1).astype(np.int32).astype("<u4").tobytes())
    got_cpp = np.frombuffer(subprocess.run([tex_dump], input=payload, capture_output=True,
                                           check=True).stdout, dtype="<u4")
    level = tex_model.words_to_bytes(words)
    got_py = tex_model.bytes_to_words(tex_model.sample_level(level, u, v, log_w, log_h, filt, wrap))
    np.testing.assert_array_equal(got_py, got_cpp)


def test_taps_stay_inside_tile():
    layout = AtlasLayout(8)
    m = np.repeat(np.arange(8), 4)
    u = np.tile([0, 1, ONE - 2, ONE - 1], 8)
    taps = compute_taps(layout, m, u, u[::-1].copy())  # asserts internally
    assert len(taps) == WORDS


def _random_trained_model(layout, material, hidden, seed):
    torch.manual_seed(seed)
    model = NTCMaterial(layout, hidden=hidden, seed=seed)
    with torch.no_grad():
        for p in model.latents:
            p.uniform_(0, 1)
    rng = np.random.default_rng(seed)
    u, v = rng.integers(0, ONE, (2, 4096))
    model.start_qat(compute_taps(layout, np.full(4096, material), u, v))
    return model


@pytest.mark.parametrize("hidden", [16, 32])
def test_qat_forward_is_the_reference_decoder(tmp_path, hidden):
    """The training graph's qat forward must equal decoding the exported files."""
    layout = AtlasLayout(3)
    models = [_random_trained_model(layout, m, hidden, 7 + m) for m in range(3)]
    write_export(str(tmp_path), layout, ["a", "b", "c"], [m.export_latents() for m in models],
                 [m.export_layers() for m in models], hidden)
    ref = ReferenceDecoder(str(tmp_path))

    rng = np.random.default_rng(3)
    n = 3000
    mat = rng.integers(0, 3, n)
    u, v = rng.integers(0, ONE, (2, n))
    u[:50] = 0
    v[50:100] = ONE - 1
    expected = ref.decode(mat, u, v)
    for m in range(3):
        sel = mat == m
        with torch.no_grad():
            got = models[m](compute_taps(layout, mat[sel], u[sel], v[sel])).numpy()
        np.testing.assert_array_equal(got.astype(np.uint8), expected[sel])
        assert np.all(got == np.round(got))


def test_mlp_forward_contract():
    layers = [{"w": np.array([[1, -1]], np.int8), "b": np.array([3], np.int32), "shift": 1},
              {"w": np.array([[2]], np.int8), "b": np.array([0], np.int32), "shift": 1}]
    # acc1 = 10 - 4 + 3 = 9 -> (9 + 1) >> 1 = 5 ; acc2 = 10 -> 5
    assert mlp_forward(np.array([[10, 4]]), layers).tolist() == [[5]]
    # negative hidden clamps to 0; rounding is floor((acc + 2^(s-1)) / 2^s)
    assert mlp_forward(np.array([[0, 20]]), layers).tolist() == [[0]]
