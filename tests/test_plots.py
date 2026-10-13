import numpy as np
import pytest

from src.utils.plots import plot_confusion_matrix, plot_frcrn_test


def test_plot_frcrn_test_writes_figure_and_tolerates_missing_scenarios(tmp_path):
    results = {
        "low_snr_pretrained": {"test": {"noise_si_snr": 1.0, "loss": 300.0}},
        "low_snr_finetuned": {"test": {"noise_si_snr": 2.0, "loss": 280.0}},
        "high_snr_pretrained": {"test": {"noise_si_snr": -14.0, "loss": 590.0}},
    }
    path = plot_frcrn_test(results, ["low_snr", "high_snr"], tmp_path / "figure" / "frcrn_test.png")
    assert path.is_file()
    assert path.stat().st_size > 0


def test_plot_confusion_matrix_writes_figure_and_rejects_bad_shape(tmp_path):
    matrix = np.array([[3, 1, 0], [0, 0, 0], [1, 0, 2]])
    path = plot_confusion_matrix(matrix, ["a", "b", "c"], tmp_path / "figure" / "confusion.png", "demo")
    assert path.is_file()
    assert path.stat().st_size > 0
    with pytest.raises(ValueError):
        plot_confusion_matrix(matrix, ["a", "b"], tmp_path / "bad.png", "demo")
