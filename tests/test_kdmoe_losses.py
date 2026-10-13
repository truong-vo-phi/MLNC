import numpy as np
import pytest
import torch

from scripts.build_group_map import build_group_map, membership_matrix
from src.loss import (
    ConsistencyLoss,
    FrontLoss,
    KDMoELoss,
    KDMoELossConfig,
    RouterKDLoss,
    load_membership,
    router_kl,
    teacher_group_probs,
)


@pytest.fixture(scope="module")
def membership() -> torch.Tensor:
    return torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
    )


def test_router_kd_zero_for_matching_group_distribution(membership):
    torch.manual_seed(0)
    teacher = torch.randn(4, 5)
    p_t = teacher_group_probs(teacher, membership, temperature=1.0)
    router_logits = p_t.clamp_min(1e-8).log()
    loss = RouterKDLoss(membership, temperature=1.0)(router_logits, teacher)
    assert loss.abs() < 1e-5


def test_router_kd_has_no_gradient_to_teacher(membership):
    student = torch.randn(4, 2, requires_grad=True)
    teacher = torch.randn(4, 5, requires_grad=True)
    RouterKDLoss(membership)(student, teacher).backward()
    assert teacher.grad is None
    assert student.grad is not None


def test_teacher_group_probs_rows_sum_to_one(membership):
    torch.manual_seed(1)
    probs = teacher_group_probs(torch.randn(8, 5), membership, temperature=2.0)
    assert probs.shape == (8, 2)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-4)


def test_router_kl_scales_with_temperature_square(membership):
    torch.manual_seed(2)
    teacher = torch.randn(4, 5)
    student = torch.zeros(4, 2, requires_grad=True)
    for temperature in (1.0, 2.0):
        student.grad = None
        router_kl(student, teacher, membership, temperature).mean().backward()
        assert student.grad is not None


def test_front_loss_penalizes_overestimate_more():
    target = torch.zeros(1, 4)
    over = FrontLoss(lam_over=2.0)(target + 0.5, target)
    under = FrontLoss(lam_over=2.0)(target - 0.5, target)
    assert over > under
    assert torch.allclose(over, 2.0 * under)


def test_consistency_loss_zero_for_identical_pairs_and_skips_singles():
    embedding = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert ConsistencyLoss()(embedding, ["a", "a", "b"]).abs() < 1e-6
    different = ConsistencyLoss()(embedding, ["a", "b", "a"])
    assert different > 0.1
    assert ConsistencyLoss()(embedding[:1], ["a"]).item() == 0.0


def test_kdmoe_loss_terms_and_total():
    torch.manual_seed(3)
    config = KDMoELossConfig()
    loss = KDMoELoss(config, load_membership())
    batch = {
        "label": torch.tensor([0, 5]),
        "group": torch.tensor([2, 1]),
        "snr_db": torch.tensor([0.0, 15.0]),
        "teacher_logits36": torch.randn(2, 36),
        "noise_ytid": ["abc", "abc"],
    }

    class Output:
        logits = torch.randn(2, 36)
        router_logits = torch.randn(2, 4)
        e = torch.randn(2, 4)
        noise_mel = torch.randn(2, 1, 128, 384)
        noise_target = torch.randn(2, 1, 128, 384)

    terms = loss(Output(), batch)
    assert set(terms) == {"ce", "kd", "front", "cons", "total"}
    expected = (
        terms["ce"]
        + config.kd_weight * terms["kd"]
        + config.cons_weight * terms["cons"]
        + config.front_weight * terms["front"]
    )
    assert torch.allclose(terms["total"], expected, atol=1e-6)
    assert all(torch.isfinite(value) for value in terms.values())


def test_kdmoe_loss_route_ce_optional():
    torch.manual_seed(4)
    config = KDMoELossConfig(route_ce_weight=0.5)
    loss = KDMoELoss(config, load_membership())
    batch = {
        "label": torch.tensor([0, 5]),
        "group": torch.tensor([2, 1]),
        "snr_db": torch.tensor([0.0, 15.0]),
        "teacher_logits36": torch.randn(2, 36),
        "noise_ytid": ["abc", "abc"],
    }

    class Output:
        logits = torch.randn(2, 36)
        router_logits = torch.randn(2, 4)
        e = torch.randn(2, 4)
        noise_mel = None
        noise_target = None

    terms = loss(Output(), batch)
    assert "route_ce" in terms and "front" not in terms


def test_group_map_sizes_and_order(tmp_path):
    group_map = build_group_map(out_dir=tmp_path)
    assert len(group_map["classes"]) == 36
    sizes = {group: len(mids) for group, mids in group_map["groups"].items()}
    assert sizes == {"E0": 5, "E1": 12, "E2": 9, "E3": 10}
    matrix = membership_matrix(group_map)
    assert matrix.shape == (36, 4)
    assert matrix.sum() == 36
    assert np.allclose(matrix.sum(axis=1), 1.0)
    assert [row["name"] for row in group_map["classes"]] == sorted(row["name"] for row in group_map["classes"])
    assert [row["label_index"] for row in group_map["classes"]] == list(range(36))


def test_load_membership_matches_generated(tmp_path):
    build_group_map(out_dir=tmp_path)
    generated = np.load(tmp_path / "membership.npy")
    assert torch.allclose(load_membership(), torch.from_numpy(generated).float())
    assert load_membership().shape == (36, 4)
    assert torch.allclose(load_membership().sum(dim=1), torch.ones(36), atol=1e-6)
