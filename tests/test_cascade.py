from pathlib import Path

import pytest
from PIL import Image

from rsdiff.cascade import SYNTHETIC_NOTICE, generate, sha256_file, verify_checkpoint
from rsdiff.eval.clip import bundle_pairs


class FakeSampler:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return [Image.new("RGB", (8, 8), (i, 0, 0)) for i, _ in enumerate(kwargs["texts"])]


def test_bundle_pairs_files_with_captions_in_order(tmp_path: Path):
    captions = [f"caption {i}" for i in range(101)]  # index width grows past 2 digits
    sampler = FakeSampler()
    generate(sampler, captions, tmp_path, batch_size=7, cond_scale=5.0)

    paired_captions, pngs = bundle_pairs(tmp_path)
    assert paired_captions == captions
    assert [Image.open(p).text["Prompt"] for p in pngs] == captions
    assert all(c["cond_scale"] == 5.0 and "stop_at_unet_number" not in c for c in sampler.calls)


def test_png_is_marked_synthetic(tmp_path: Path):
    (path,) = generate(FakeSampler(), ["a dense residential area"], tmp_path)
    text = Image.open(path).text
    assert text["Description"] == SYNTHETIC_NOTICE
    assert text["Software"].startswith("rsdiff ")


def test_resume_only_samples_incomplete_batches(tmp_path: Path):
    captions = ["a", "b", "c", "d", "e"]
    paths = generate(FakeSampler(), captions, tmp_path, batch_size=2)
    paths[3].unlink()  # second batch (c, d) incomplete

    sampler = FakeSampler()
    generate(sampler, captions, tmp_path, batch_size=2, stop_at_unet=1)
    assert [c["texts"] for c in sampler.calls] == [["c", "d"]]
    assert sampler.calls[0]["stop_at_unet_number"] == 1
    assert all(p.exists() for p in paths)


@pytest.mark.parametrize("captions", [[], ["ok", "  "]])
def test_empty_captions_are_rejected(tmp_path: Path, captions):
    with pytest.raises(ValueError):
        generate(FakeSampler(), captions, tmp_path)


def test_bundle_pairs_rejects_count_mismatch_and_ignores_grid(tmp_path: Path):
    generate(FakeSampler(), ["a", "b"], tmp_path)
    Image.new("RGB", (8, 8)).save(tmp_path / "_grid.png")
    assert len(bundle_pairs(tmp_path)[1]) == 2

    (tmp_path / "99_extra.png").write_bytes((tmp_path / "_grid.png").read_bytes())
    with pytest.raises(ValueError, match="3 PNGs but 2 captions"):
        bundle_pairs(tmp_path)


def test_verify_checkpoint(tmp_path: Path):
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"weights")
    verify_checkpoint(ckpt, sha256_file(ckpt))
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_checkpoint(ckpt, "0" * 64)
