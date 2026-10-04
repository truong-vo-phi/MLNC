from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.load_dataset import (
    AUDIOSET_KEY,
    MIXTURE_KEY,
    AudioSet40Dataset,
    Mixture36Dataset,
    fit_length,
    load_waveform,
    mix_at_snr,
    resolve_dataset_root,
    subsample,
)


def test_fit_length_pads_and_crops():
    assert fit_length(np.ones(5), 8).tolist() == [1, 1, 1, 1, 1, 0, 0, 0]
    assert fit_length(np.ones(10), 4).shape == (4,)


@pytest.mark.parametrize("snr_db", [-5.0, 0.0, 20.0])
def test_mix_at_snr_hits_target(snr_db):
    rng = np.random.default_rng(0)
    speech = rng.normal(size=16000).astype(np.float32) * 0.1
    noise = rng.normal(size=16000).astype(np.float32)
    mixture, scaled = mix_at_snr(speech, noise, snr_db)
    residual = mixture - scaled
    measured = 10 * np.log10(np.mean(residual**2) / np.mean(scaled**2))
    assert measured == pytest.approx(snr_db, abs=0.1)


def test_subsample_is_evenly_spaced():
    sub = subsample(_Sized(100), 5)
    assert len(sub) == 5
    assert sub.indices[0] == 0
    assert sub.indices[-1] == 99


class _Sized:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return index


def test_resolve_dataset_root_validates(tmp_path):
    config = tmp_path / "url.yaml"
    config.write_text(f"present: '{tmp_path}'\nempty:\nmissing: '{tmp_path / 'nope'}'\n", encoding="utf-8")
    assert resolve_dataset_root("present", config) == Path(tmp_path)
    with pytest.raises(KeyError):
        resolve_dataset_root("empty", config)
    with pytest.raises(KeyError):
        resolve_dataset_root("absent", config)
    with pytest.raises(FileNotFoundError):
        resolve_dataset_root("missing", config)


def test_load_waveform_resamples(tmp_path):
    path = tmp_path / "tone.wav"
    sf.write(path, np.random.default_rng(0).normal(size=(44100 * 2, 2)).astype(np.float32) * 0.1, 44100)
    segment = load_waveform(path, 16000, 16000, np.random.default_rng(1))
    assert segment.shape == (16000,)


def _dataset_available(key):
    try:
        resolve_dataset_root(key)
    except (KeyError, FileNotFoundError):
        return False
    return True


@pytest.mark.data
@pytest.mark.skipif(not _dataset_available(AUDIOSET_KEY), reason="AudioSet-40l not available")
def test_audioset_item():
    dataset = AudioSet40Dataset(split="validation")
    item = dataset[0]
    assert item["waveform"].shape == (160000,)
    assert item["target"].shape == (dataset.num_classes,)
    assert item["target"].sum() >= 1


@pytest.mark.data
@pytest.mark.skipif(not _dataset_available(MIXTURE_KEY), reason="Mixture-36l not available")
def test_mixture_item_and_snr_filter():
    dataset = Mixture36Dataset(split="validation", snr_db=(15.0, 20.0))
    item = dataset[0]
    assert item["waveform"].shape == (64000,)
    assert item["target"].sum() == 1
    assert float(item["snr_db"]) in (15.0, 20.0)


def test_audioset_uses_fallback_labels(tmp_path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (tmp_path / "audio").mkdir()
    sf.write(tmp_path / "audio" / "abc.wav", np.zeros(16000, dtype=np.float32), 16000)
    header = '"split","ytid","status","label_mids"'
    row = '"train","abc","copied","/m/032s66"'
    (metadata / "manifest.csv").write_text("\n".join([header, row]) + "\n", encoding="utf-8")
    dataset = AudioSet40Dataset(root=tmp_path, split="train", duration=1.0)
    assert dataset.num_classes == 40
    assert len(dataset) == 1
    assert dataset[0]["target"].sum() == 1
