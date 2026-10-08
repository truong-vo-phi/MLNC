from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.download_dataset import DATASETS
from scripts.load_dataset import (
    ALL_SNR,
    AUDIOSET_KEY,
    MIXTURE_KEY,
    VLSP_KEY,
    AudioSet40Dataset,
    Mixture36Dataset,
    VLSPDataset,
    VLSPMixDataset,
    build_dataset,
    dataset_env_name,
    fit_length,
    load_waveform,
    mix_at_snr,
    read_class_mids,
    read_split_ratios,
    resolve_dataset_root,
    split_of,
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


def test_read_class_mids_follows_given_order():
    assert read_class_mids(["Laughter", "Gunshot, gunfire"]) == ["/m/01j3sz", "/m/032s66"]
    with pytest.raises(KeyError):
        read_class_mids(["not a class"])


class _StubNoise:
    def __init__(self, path):
        self.samples = [(path, None)]

    def __len__(self):
        return 1


def _write_tone(path, seconds=1.0, rate=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * rate)) / rate
    sf.write(path, (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), rate)


def test_split_of_is_deterministic_and_follows_ratios():
    ratios = {"train": 0.5, "validation": 0.25, "test": 0.25}
    names = [f"speaker/utt_{i}.wav" for i in range(2000)]
    splits = [split_of(name, ratios) for name in names]
    assert splits == [split_of(name, ratios) for name in names]
    for split, ratio in ratios.items():
        assert splits.count(split) / len(names) == pytest.approx(ratio, abs=0.05)


def test_read_split_ratios_requires_sum_one(tmp_path):
    path = tmp_path / "vlsp.yaml"
    path.write_text("split:\n  train: 0.9\n  validation: 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_split_ratios(path)
    path.write_text("split:\n  train: 0.8\n  validation: 0.2\n", encoding="utf-8")
    assert read_split_ratios(path) == {"train": 0.8, "validation": 0.2}


def test_vlsp_mix_dataset_partitions_audio_files(tmp_path):
    for index in range(30):
        _write_tone(tmp_path / f"speaker_{index % 3}" / f"utt_{index}.wav", seconds=1.5)
    (tmp_path / "transcript.txt").write_text("not audio", encoding="utf-8")
    _write_tone(tmp_path / "noise.wav")
    ratios = {"train": 0.6, "validation": 0.2, "test": 0.2}
    kwargs = {"root": tmp_path, "noise_dataset": _StubNoise(tmp_path / "noise.wav"), "split_ratios": ratios}
    parts = {split: VLSPMixDataset(split=split, duration=1.0, **kwargs) for split in ratios}
    paths = [set(part.speech_paths) for part in parts.values()]
    assert sum(len(p) for p in paths) == 31
    assert set.union(*paths) == {*tmp_path.rglob("*.wav")}
    assert all(not a & b for i, a in enumerate(paths) for b in paths[i + 1 :])
    item = parts["train"][0]
    assert item["waveform"].shape == item["noise"].shape == (16000,)
    assert float(item["snr_db"]) in ALL_SNR


def test_vlsp_mix_dataset_validates_inputs(tmp_path):
    stub = _StubNoise(tmp_path / "noise.wav")
    with pytest.raises(FileNotFoundError):
        VLSPMixDataset(root=tmp_path, split="train", noise_dataset=stub, split_ratios={"train": 1.0})
    _write_tone(tmp_path / "utt.wav")
    with pytest.raises(ValueError):
        VLSPMixDataset(root=tmp_path, split="dev", noise_dataset=stub, split_ratios={"train": 1.0})


def test_vlsp_dataset_is_registered_without_marker():
    spec = DATASETS[VLSP_KEY]
    assert spec.kaggle_key == "kaggle-vlsp"
    assert spec.marker == ""
    assert dataset_env_name(VLSP_KEY) == "MLNC_VLSP_100H"


def test_mix_at_snr_ignores_padding_in_speech_power():
    rng = np.random.default_rng(0)
    speech = np.pad(rng.normal(size=8000).astype(np.float32) * 0.1, (0, 8000))
    noise = rng.normal(size=16000).astype(np.float32)
    mixture, scaled = mix_at_snr(speech, noise, 5.0, speech_length=8000)
    measured = 10 * np.log10(np.mean((mixture - scaled)[:8000] ** 2) / np.mean(scaled**2))
    assert measured == pytest.approx(5.0, abs=0.1)
    _, diluted = mix_at_snr(speech, noise, 5.0)
    assert np.mean(diluted**2) < np.mean(scaled**2)


@pytest.mark.parametrize("seconds", [0.5, 2.0])
def test_vlsp_mix_snr_uses_only_real_speech(tmp_path, seconds):
    _write_tone(tmp_path / "speech" / "utt.wav", seconds=seconds)
    _write_tone(tmp_path / "noise.wav", seconds=2.0)
    dataset = VLSPMixDataset(
        root=tmp_path / "speech",
        split="train",
        noise_dataset=_StubNoise(tmp_path / "noise.wav"),
        split_ratios={"train": 1.0},
        snr_levels=(5.0,),
        duration=1.0,
    )
    item = dataset[0]
    valid = min(int(seconds * 16000), 16000)
    speech = (item["waveform"] - item["noise"]).numpy()
    measured = 10 * np.log10(np.mean(speech[:valid] ** 2) / np.mean(item["noise"].numpy() ** 2))
    assert measured == pytest.approx(5.0, abs=0.1)


def test_vlsp_dataset_is_a_standalone_audio_pool(tmp_path):
    for index in range(20):
        _write_tone(tmp_path / f"speaker_{index % 2}" / f"utt_{index}.wav", seconds=1.5)
    ratios = {"train": 0.5, "validation": 0.25, "test": 0.25}
    parts = {split: VLSPDataset(tmp_path, split, duration=1.0, split_ratios=ratios) for split in ratios}
    assert sum(len(part) for part in parts.values()) == 20
    path = parts["train"].paths[0]
    assert path.is_file()
    item = parts["train"][0]
    assert item["waveform"].shape == (16000,)
    assert item["sample_id"] == path.relative_to(tmp_path).as_posix()


def test_vlsp_dataset_validates_split(tmp_path):
    _write_tone(tmp_path / "utt.wav")
    with pytest.raises(ValueError):
        VLSPDataset(tmp_path, "dev", split_ratios={"train": 1.0})


def test_build_dataset_registers_vlsp(tmp_path, monkeypatch):
    _write_tone(tmp_path / "utt.wav")
    monkeypatch.setenv(dataset_env_name(VLSP_KEY), str(tmp_path))
    dataset = build_dataset(VLSP_KEY, "train", split_ratios={"train": 1.0})
    assert isinstance(dataset, VLSPDataset)
    assert len(dataset) == 1
