import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opera.metrics import compute_eer, compute_det_curve, ctrsvdd_metrics, track_level_aggregate, evaluate_all

# 需要 torch 的用例在无 torch 环境下跳过（协议层用例如常执行）
requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed")


def test_threshold_endpoints_ties_and_direction():
    assert compute_eer([-2, -1, 1, 2], [0, 0, 1, 1])[0] == 0
    assert compute_eer([2, 1, -1, -2], [0, 0, 1, 1])[0] == 1
    assert compute_eer([0, 0, 0, 0], [0, 0, 1, 1])[0] == .5
    far, frr, th = compute_det_curve([0, 0, 1, 2], [1, 0, 0, 1])
    assert len(th) == 4
    assert (far[0], frr[0], far[-1], frr[-1]) == (0, 1, 1, 0)
    for a, b, t in zip(far, frr, th):
        s, y = np.array([0, 0, 1, 2]), np.array([1, 0, 0, 1])
        assert a == np.mean(s[y == 0] >= t)
        assert b == np.mean(s[y == 1] < t)


@pytest.mark.parametrize("s,y", [([], []), ([1], [1]), ([0, 1], [0, 2]), ([np.nan, 1], [0, 1]), ([0, 1], [0])])
def test_invalid_scores_fail(s, y):
    with pytest.raises(ValueError):
        compute_eer(s, y)


def test_paper_cohort_keeps_all_bonafide():
    """论文 Table 1 口径：只排除 A14 的 spoof，保留 ACESinger 的 bonafide。

    依据 scripts/reproduce_table1_rows.py 的实测：在该口径下用官方 B01/B02
    分数重算得到 pooled EER 11.37% / 10.39%，与论文 Table 1 一致；若额外剔除
    acesinger bonafide 则会得到 12.03% / 11.16%，比论文高 0.7–0.8 个百分点。
    """
    m = ctrsvdd_metrics([-2, -1, 1, 2, 9, -9], [0, 0, 1, 1, 0, 1],
                        ['-', '-', 'A09', 'A10', '-', 'A14'],
                        ['kising', 'm4singer', 'kising', 'm4singer', 'acesinger', 'acesinger'])
    assert m['protocol'] == 'ctrsvdd_a09_a13_bonafide_all'
    assert m['n_bonafide'] == 3          # 含 acesinger 那 1 条 bonafide
    assert m['n_spoof'] == 2             # 只有 A09/A10，A14 被排除
    assert m['by_attack']['A09']['n_bonafide'] == 3


def test_legacy_cohort_excludes_acesinger():
    """与 CtrSVDD2024_Baseline/analysis/results.csv 对账时使用的旧口径。"""
    m = ctrsvdd_metrics([-2, -1, 1, 2, 9, -9], [0, 0, 1, 1, 0, 1],
                        ['-', '-', 'A09', 'A10', '-', 'A14'],
                        ['kising', 'm4singer', 'kising', 'm4singer', 'acesinger', 'acesinger'],
                        bonafide_cohort='non_acesinger')
    assert m['protocol'] == 'ctrsvdd_a09_a13_bonafide_non_acesinger'
    assert m['n_bonafide'] == 2
    assert m['n_spoof'] == 2
    assert m['EER_%'] == 0


def test_unknown_cohort_rejected():
    with pytest.raises(ValueError, match='unknown bonafide_cohort'):
        ctrsvdd_metrics([-2, 1], [0, 1], ['-', 'A09'], ['kising', 'kising'],
                        bonafide_cohort='whatever')


def test_track_aggregation_rejects_conflicting_labels():
    with pytest.raises(ValueError):
        track_level_aggregate([1, 2], ['same', 'same'], [0, 1])
    s, y = track_level_aggregate([1, 3, 5], ['T02/a', 'T02/a', 'T03/a'], [0, 0, 1])
    np.testing.assert_equal(s, [2, 5])
    np.testing.assert_equal(y, [0, 1])


def test_current_metrics_do_not_substitute_dcf():
    assert 'min_tDCF' not in evaluate_all([-1, 1], [0, 1])


@requires_torch
def test_missing_t03_is_not_silently_omitted(tmp_path):
    from opera.data.singfake import scan_singfake_tracks
    (tmp_path/'bonafide').mkdir()
    (tmp_path/'spoof').mkdir()
    with pytest.raises(FileNotFoundError):
        scan_singfake_tracks(tmp_path, ['T03'])


@requires_torch
def test_training_never_selects_on_test():
    from train import build_loaders
    with pytest.raises(ValueError, match='test is reserved'):
        build_loaders({'data': {'train': {'splits': ['train']}, 'eval': {'splits': ['test']}}})


@requires_torch
def test_wrap_phase_boundary():
    import torch
    from opera.cqt import wrap_to_pi
    y = wrap_to_pi(torch.tensor([-torch.pi, 0, torch.pi, 3*torch.pi], dtype=torch.float64))
    torch.testing.assert_close(y, torch.tensor([torch.pi, 0, torch.pi, torch.pi],dtype=torch.float64))


@requires_torch
def test_resnet_keeps_time_after_stem():
    import torch
    from opera.models.resnet18 import ResNet18Encoder
    model=ResNet18Encoder(base_channels=4,feat_dim=8).eval()
    with torch.no_grad():
        y=model(torch.zeros(2,2,84,200))
    assert y.shape==(2,8,100)
