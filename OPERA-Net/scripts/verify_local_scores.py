"""Recompute existing B01/B02 scores. These are baseline runs, not OPERA-Net runs.

注意：官方 results.csv 的 "EER (w/o A14)" 用的是 `bonafide_cohort='non_acesinger'`
口径（额外剔除 acesinger 的 2,845 条 bonafide）。论文 Table 1 用的是默认的
`'all'` 口径。本脚本的目的就是与 results.csv 对账，故显式指定旧口径；
论文口径的复现见 scripts/reproduce_table1_rows.py。
"""
import csv
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opera.metrics import ctrsvdd_metrics


def main():
    base = ROOT.parent / 'CtrSVDD2024_Baseline/analysis'
    truth = {r[2]: r for line in (base/'test_groundtruth.txt').read_text().splitlines() if (r := line.split())}
    ref = {r['csv']: r for r in csv.DictReader((base/'results.csv').open())}
    results = []
    for name in ['B01', 'B02']:
        rows = list(csv.DictReader((base/f'baselines_csv/{name}.csv').open()))
        ids = [r['filename'] for r in rows]
        if len(set(ids)) != len(ids) or set(ids) != set(truth):
            raise ValueError('Scores must cover each ground-truth utterance exactly once')
        m = ctrsvdd_metrics([-float(r['score']) for r in rows],
                            [int(truth[u][5] != 'bonafide') for u in ids],
                            [truth[u][4] for u in ids], [truth[u][0] for u in ids],
                            bonafide_cohort='non_acesinger')
        prior = ref[name+'.csv']
        delta = m['EER_%'] - 100*float(prior['EER (w/o A14)'])
        results.append(dict(system=name, source=f'CtrSVDD2024_Baseline/analysis/baselines_csv/{name}.csv',
                            note='existing baseline score file; not paper OPERA-Net evidence',
                            original_score_direction='higher_is_bonafide; negated for current API',
                            recomputed=m, saved_results_eer_percent=100*float(prior['EER (w/o A14)']),
                            delta_percentage_points=delta))
        # Ties may differ from the official sequential tie convention by one observation.
        assert abs(delta) < .01, (name, delta)
    out=ROOT.parent/'restored_data/current_pdf/local_baseline_recomputed.json'
    out.write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps([{r['system']:r['recomputed']['EER_%'],'delta':r['delta_percentage_points']} for r in results]))


if __name__ == '__main__':
    main()
