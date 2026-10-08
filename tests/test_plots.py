from src.utils.plots import plot_frcrn_test


def test_plot_frcrn_test_writes_figure_and_tolerates_missing_scenarios(tmp_path):
    results = {
        "low_snr_pretrained": {"test": {"noise_si_snr": 1.0, "loss": 300.0}},
        "low_snr_finetuned": {"test": {"noise_si_snr": 2.0, "loss": 280.0}},
        "high_snr_pretrained": {"test": {"noise_si_snr": -14.0, "loss": 590.0}},
    }
    path = plot_frcrn_test(results, ["low_snr", "high_snr"], tmp_path / "figure" / "frcrn_test.png")
    assert path.is_file()
    assert path.stat().st_size > 0
