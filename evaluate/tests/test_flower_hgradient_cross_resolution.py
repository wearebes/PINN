from evaluate.flower_hgradient_cross_resolution import _summarize_family_step_curves


def test_summarize_family_step_curves_uses_pair_distribution_per_step():
    rows = [
        {"family": "smooth", "train_rho": 64, "test_rho": 64, "iter": 1, "numeric_mse": 1.0, "model_mse": 2.0},
        {"family": "smooth", "train_rho": 128, "test_rho": 64, "iter": 1, "numeric_mse": 3.0, "model_mse": 6.0},
        {"family": "smooth", "train_rho": 64, "test_rho": 64, "iter": 2, "numeric_mse": 10.0, "model_mse": 20.0},
        {"family": "smooth", "train_rho": 128, "test_rho": 64, "iter": 2, "numeric_mse": 30.0, "model_mse": 60.0},
        {"family": "acute", "train_rho": 64, "test_rho": 64, "iter": 1, "numeric_mse": 100.0, "model_mse": 200.0},
    ]

    summary = _summarize_family_step_curves(rows, "smooth")

    assert summary["iter"].tolist() == [1, 2]
    assert summary["pair_count"].tolist() == [2, 2]
    assert summary["numeric_median"].tolist() == [2.0, 20.0]
    assert summary["model_median"].tolist() == [4.0, 40.0]
    assert summary["numeric_q25"].tolist() == [1.5, 15.0]
    assert summary["numeric_q75"].tolist() == [2.5, 25.0]
    assert summary["model_q25"].tolist() == [3.0, 30.0]
    assert summary["model_q75"].tolist() == [5.0, 50.0]


def test_summarize_family_step_curves_can_filter_to_one_resolution_pair():
    rows = [
        {"family": "acute", "train_rho": 128, "test_rho": 64, "iter": 1, "numeric_mse": 1.0, "model_mse": 10.0},
        {"family": "acute", "train_rho": 256, "test_rho": 64, "iter": 1, "numeric_mse": 2.0, "model_mse": 20.0},
        {"family": "acute", "train_rho": 256, "test_rho": 128, "iter": 1, "numeric_mse": 4.0, "model_mse": 40.0},
        {"family": "acute", "train_rho": 256, "test_rho": 256, "iter": 1, "numeric_mse": 8.0, "model_mse": 80.0},
    ]

    summary = _summarize_family_step_curves(rows, "acute", train_rho=256, test_rho=256)

    assert summary["iter"].tolist() == [1]
    assert summary["pair_count"].tolist() == [1]
    assert summary["numeric_median"].tolist() == [8.0]
    assert summary["model_median"].tolist() == [80.0]
