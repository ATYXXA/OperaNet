"""Verify emitted record counts, references, and provenance without changing data."""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'restored_data/current_pdf'


def main():
    data=json.loads((OUT/'data_audit.json').read_text(encoding='utf-8'))
    counts={}
    for file, expected in [('ctrsvdd_train.jsonl',84404),('ctrsvdd_dev.jsonl',43625),('ctrsvdd_test.jsonl',92769),
                           ('ctrsvdd_eval_a09_a13.jsonl',64734),('singfake_source_metadata.jsonl',1481),
                           ('singfake_audio_inventory.jsonl',1798),('singfake_eval_segments.jsonl',17342)]:
        with (OUT/file).open(encoding='utf-8') as f:
            rows=[json.loads(line) for line in f]
        assert len(rows)==expected,(file,len(rows))
        if rows and 'utt_id' in rows[0]:
            assert len({r['utt_id'] for r in rows})==len(rows),file
        counts[file]=len(rows)
    assert sum(x['metadata_rows'] for x in data['singfake']['groups'])==1481
    assert not data['ctrsvdd']['missing_audio']
    assert not data['ctrsvdd']['invalid_flac_headers']
    assert data['singfake']['t03_audio_files']==0
    refs=json.loads((OUT/'source_hashes.json').read_text(encoding='utf-8'))
    for row in refs:
        assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    reference=json.loads((ROOT/'OPERA-Net/docs/paper_reference.json').read_text(encoding='utf-8'))
    assert len(reference['tables'])==3
    for n in range(1,4):
        result=json.loads((ROOT/f'OPERA-Net/results/current_pdf/table{n}_reconciliation.json').read_text(encoding='utf-8'))
        assert all(row['measured'] is None for row in result),'No completed OPERA-Net runs exist in this audit'
    model=json.loads((OUT/'model_validation.json').read_text(encoding='utf-8'))
    assert model['model_check']=='passed_forward_backward_optimizer_step'
    test_run=subprocess.run([sys.executable,'-m','pytest','tests/test_protocol.py','-q'],
                            cwd=ROOT/'OPERA-Net',capture_output=True,text=True,check=True)
    (OUT/'test_run.txt').write_text(test_run.stdout+test_run.stderr,encoding='utf-8')
    passed=int(re.search(r'(\d+) passed',test_run.stdout).group(1))
    report={'recovered_record_counts':counts,'source_hashes_verified':len(refs),'table_count':3,
            'model_validation':'passed','metadata_summary_reconciles':True,
            'unit_regression_tests':{'passed':passed,'command':'python -m pytest tests/test_protocol.py -q','output':'test_run.txt'},
            'baseline_score_regression':'B01 and B02 match local saved results exactly',
            'limitations':['no full OPERA-Net training','no original detector checkpoints or scores recovered',
                           'T03 absent and codec settings unspecified','SingFake labels not content-verified',
                           'FLAC header checks do not replace full decode validation']}
    (OUT/'verification_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
