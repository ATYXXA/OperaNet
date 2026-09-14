"""Build a provenance-labelled candidate experiment, not original paper artifacts.

prepare: exact available-file center-window trials plus two explicit T03 hypotheses.
run: 32-example pilot training, resumable checkpoint, and real scores for every trial.
verify: checkpoint reload, trial/score coverage, file hashes, and score replay.
No paper metric values are used by training, selection, scoring or verification.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / 'OPERA-Net'
AUDIT = ROOT / 'restored_data/current_pdf'
sys.path.insert(0, str(CODE))
from opera.models import build_model
from opera.metrics import evaluate_all
from opera.utils import set_seed

STATUS = 'candidate_reproduction_pilot_not_original_paper'
CODECS = [dict(id='vorbis_64k', encoder='libvorbis', bitrate='64k', extension='ogg', options=[]),
          dict(id='opus_64k', encoder='libopus', bitrate='64k', extension='opus',
               options=['-vbr', 'on', '-application', 'audio', '-frame_duration', '20'])]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')


def readl(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def fix(y):
    return np.pad(y[:64000],(0,max(0,64000-len(y)))).astype('float32')


def extract_center(source,output,random_seed=None):
    import librosa
    with sf.SoundFile(str(source)) as f:
        n=min(f.frames,4*f.samplerate)
        start=(f.frames-n)//2 if random_seed is None else random.Random(random_seed).randint(0,f.frames-n)
        rate=f.samplerate
        f.seek(start)
        y=f.read(n,dtype='float32',always_2d=True).mean(axis=1)
    if rate!=16000:
        y=librosa.resample(y,orig_sr=rate,target_sr=16000,res_type='soxr_hq')
    output.parent.mkdir(parents=True,exist_ok=True)
    sf.write(str(output),fix(y),16000,format='FLAC',subtype='PCM_24')
    return dict(source_start_sample=start,source_end_sample=start+n,source_sample_rate=rate,
                start_seconds=start/rate,end_seconds=(start+n)/rate,source_sha256=sha(source),
                preprocessing='seek native samples; mean channels; soxr_hq to 16k; crop/right-pad to 64000; PCM_24 FLAC',
                output_sha256=sha(output))


def select_balanced(records,n_per_class,seed):
    result=[]
    for label in [0,1]:
        candidates=[r for r in records if r['label_id']==label]
        candidates.sort(key=lambda r:hashlib.sha256(f"{seed}:{r['utt_id']}".encode()).hexdigest())
        result.extend(candidates[:n_per_class])
    return result


def prepare(out):
    if (out/'prepared.json').exists():
        raise FileExistsError('Use another --out for a new preparation; existing trials remain immutable')
    import imageio_ffmpeg
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    version=subprocess.run([ffmpeg,'-version'],capture_output=True,text=True,check=True).stdout
    cfg=yaml.safe_load((CODE/'configs/opera_net.yaml').read_text(encoding='utf-8'))
    proposed=copy.deepcopy(cfg)
    proposed['paper']['status']='hypothesized_full_run_not_executed'
    proposed['protocol_note']='Current source implementation defaults are hypotheses, not recovered author hyperparameters.'
    save(out/'hypothesized_full_training.json',proposed)
    cfg.update(device='cpu',epochs=1,train_batch_size=2,eval_batch_size=4,num_workers=0,amp=False,seed=42)
    cfg['optim'].update(warmup_ratio=0,min_lr_ratio=1)
    cfg['pilot']={'status':STATUS,'train_examples':32,'dev_examples':16,'epochs':1,'scheduler':'constant',
                  'data_selection':'lowest sha256(seed:utt_id) per label; never selected by predictions',
                  'sampling':'balanced subset; one reproducible crop per training example',
                  'checkpoint_selection':'final after predeclared 16 optimizer steps; no test selection',
                  'cross_eval':'available local T01/T02 files, one center window each; derived codecs from T02 windows',
                  'metrics_comparable_to_paper':False}
    save(out/'pilot_config.json',cfg)
    codec_config={'status':'explicit_candidate_hypothesis_not_author_configuration','source_subset':'T02',
                  'source_unit':'reconstructed center 4-second window from each available file',
                  'codecs':CODECS,'upstream_example':'https://raw.githubusercontent.com/yongyizang/SingFake/main/dataset/simulate_codec.py',
                  'upstream_evidence':'Example lists ogg 64k and opus 64k; it does not identify the author experiment settings.',
                  'ffmpeg_path':ffmpeg,'ffmpeg_version':version,'output_sample_rate':16000,'output_channels':1,
                  'output_samples':64000,'output_format':'FLAC PCM_24','order':'segment then encode/decode',
                  'not_equivalent_to':'codec conversion of original full tracks followed by segmentation'}
    save(out/'t03_codec_config.json',codec_config)
    records=[r for r in readl(AUDIT/'singfake_audio_inventory.jsonl') if r['subset'] in ('T01','T02')]
    trials=[]
    for i,r in enumerate(records):
        uid='SF-'+r['subset']+'-'+hashlib.sha256(r['file_id'].encode()).hexdigest()[:16]
        dst=out/'audio'/r['subset']/(uid+'.flac')
        desc=extract_center(ROOT/r['path'],dst)
        trial=dict(trial_id=uid,subset=r['subset'],label_id=r['label_id'],source_path=r['path'],
                   source_file_id=r['file_id'],audio_path=dst.relative_to(out).as_posix(),
                   metadata_candidate_lines=r['metadata_candidate_lines'],metadata_link_status=r['metadata_link_status'],
                   label_provenance='local_directory_not_content_verified',status=STATUS,codec_id=None,parent_trial_id=None,**desc)
        trials.append(trial)
        if (i+1)%50==0:
            print(f'prepared base {i+1}/{len(records)}',flush=True)
    originals=list(trials)
    codec_commands=[]
    for r in originals:
        if r['subset']!='T02':
            continue
        for codec in CODECS:
            uid=r['trial_id'].replace('SF-T02-','SF-T03-')+'-'+codec['id']
            compressed=out/'encoded'/codec['id']/(uid+'.'+codec['extension'])
            target=out/'audio'/'T03'/codec['id']/(uid+'.flac')
            compressed.parent.mkdir(parents=True,exist_ok=True)
            target.parent.mkdir(parents=True,exist_ok=True)
            encode=[ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-n','-i',str(out/r['audio_path']),
                    '-vn','-ac','1','-c:a',codec['encoder'],'-b:a',codec['bitrate'],*codec['options'],str(compressed)]
            decode=[ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-n','-i',str(compressed),
                    '-ac','1','-ar','16000','-c:a','flac','-sample_fmt','s32',str(target)]
            subprocess.run(encode,check=True,capture_output=True)
            subprocess.run(decode,check=True,capture_output=True)
            y,rate=sf.read(str(target),dtype='float32')
            assert rate==16000
            decoded_frames=len(y)
            sf.write(str(target),fix(y),16000,format='FLAC',subtype='PCM_24')
            derived=dict(r,trial_id=uid,subset='T03',audio_path=target.relative_to(out).as_posix(),
                         output_sha256=sha(target),codec_id=codec['id'],parent_trial_id=r['trial_id'],
                         compressed_path=compressed.relative_to(out).as_posix(),compressed_sha256=sha(compressed),
                         codec_decoded_samples=decoded_frames,codec_config_sha256=sha(out/'t03_codec_config.json'))
            trials.append(derived)
            codec_commands.append(dict(trial_id=uid,encode=encode,decode=decode))
        if len(codec_commands)%100==0:
            print(f'prepared codec variants {len(codec_commands)}/394',flush=True)
    assert len({r['trial_id'] for r in trials})==len(trials)
    jsonl(out/'singfake_trials.jsonl',trials)
    jsonl(out/'codec_commands.jsonl',codec_commands)
    pilot_lists={}
    for split,count in [('train',16),('dev',8)]:
        chosen=select_balanced(readl(AUDIT/f'ctrsvdd_{split}.jsonl'),count,42)
        new=[]
        for r in chosen:
            dst=out/'audio'/('pilot_'+split)/(r['utt_id']+'.flac')
            cropseed=int(hashlib.sha256(('crop42:'+r['utt_id']).encode()).hexdigest()[:16],16) if split=='train' else None
            desc=extract_center(ROOT/r['path'],dst,random_seed=cropseed)
            new.append(dict(trial_id=r['utt_id'],subset=split,label_id=r['label_id'],source_path=r['path'],
                            attack=r['attack'],singer_id=r['singer_id'],audio_path=dst.relative_to(out).as_posix(),
                            status=STATUS,**desc))
        pilot_lists[split]=new
        jsonl(out/f'pilot_{split}_trials.jsonl',new)
    assert not ({r['singer_id'] for r in pilot_lists['train']} & {r['singer_id'] for r in pilot_lists['dev']})
    save(out/'prepared.json',{'status':STATUS,'created_utc':datetime.now(timezone.utc).isoformat(),
                            'trial_counts':dict(Counter(r['subset'] for r in trials)),
                            'class_counts':dict(Counter(f"{r['subset']}:{r['label_id']}" for r in trials)),
                            'singfake_trial_sha256':sha(out/'singfake_trials.jsonl'),
                            'pilot_train_trial_sha256':sha(out/'pilot_train_trials.jsonl'),
                            'pilot_dev_trial_sha256':sha(out/'pilot_dev_trials.jsonl'),
                            'full_window_alternative':str(AUDIT/'singfake_eval_segments.jsonl'),
                            'full_window_alternative_sha256':sha(AUDIT/'singfake_eval_segments.jsonl')})
    print('prepared',len(trials),'SingFake trials',flush=True)


def load_batch(out,rows):
    arrays=[]
    for r in rows:
        y,sr=sf.read(str(out/r['audio_path']),dtype='float32')
        assert sr==16000 and y.shape==(64000,)
        arrays.append(torch.from_numpy(y))
    return torch.stack(arrays),torch.tensor([r['label_id'] for r in rows],dtype=torch.long)


def score(model,out,trials,checkpoint_hash,name):
    scores=[]
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(trials),4):
            batch=trials[start:start+4]
            x,y=load_batch(out,batch)
            logits=model(x)
            values=(logits[:,1]-logits[:,0]).tolist()
            assert np.isfinite(values).all()
            for r,v,z in zip(batch,values,logits.tolist()):
                scores.append(dict(trial_id=r['trial_id'],subset=r['subset'],label_id=r['label_id'],score=v,
                                   logits=z,codec_id=r.get('codec_id'),checkpoint_sha256=checkpoint_hash,
                                   audio_sha256=r['output_sha256'],status='actual_pilot_inference_not_paper_result'))
            if (start+4)%80==0:
                print(f'{name}: scored {min(start+4,len(trials))}/{len(trials)}',flush=True)
    jsonl(out/name,scores)
    return scores


def run(out):
    ckpt_path=out/'pilot_not_paper.pth'
    if ckpt_path.exists():
        raise FileExistsError('Checkpoint already exists; use verify or choose a fresh output directory')
    cfg=json.loads((out/'pilot_config.json').read_text())
    set_seed(cfg['seed'])
    torch.set_num_threads(4)
    model=build_model('opera_net',cfg['model']['params'])
    groups=model.param_groups(lr_head=float(cfg['optim']['lr_head']),lr_ssl=float(cfg['optim']['lr_ssl']),
                              ssl_decay=float(cfg['optim']['ssl_lr_decay']),weight_decay=float(cfg['optim']['weight_decay']))
    params=[id(p) for g in groups for p in g['params']]
    assert len(params)==len(set(params)) and set(params)=={id(p) for p in model.parameters() if p.requires_grad}
    optimizer=torch.optim.AdamW(groups,betas=tuple(cfg['optim']['betas']),eps=float(cfg['optim']['eps']))
    train=readl(out/'pilot_train_trials.jsonl')
    order=list(range(len(train)))
    random.Random(cfg['seed']).shuffle(order)
    logs=[]
    started=time.time()
    model.train()
    for start in range(0,len(order),2):
        selected=[train[i] for i in order[start:start+2]]
        x,y=load_batch(out,selected)
        optimizer.zero_grad(set_to_none=True)
        logits=model(x)
        # Balanced pilot gives inverse-frequency class weights [1,1].
        loss=torch.nn.functional.cross_entropy(logits,y,weight=torch.ones(2))
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        optimizer.step()
        logs.append(dict(step=len(logs)+1,trial_ids=[r['trial_id'] for r in selected],loss=loss.item(),
                         gradient_norm_before_clip=grad.item(),elapsed_seconds=time.time()-started,status=STATUS))
        print(f"pilot train step {len(logs)}/16 loss={loss.item():.5f}",flush=True)
    jsonl(out/'training_log.jsonl',logs)
    provenance={'status':STATUS,'original_paper_checkpoint':False,'trained_epochs':1,'optimizer_steps':len(logs),
                'train_examples':len(train),'created_utc':datetime.now(timezone.utc).isoformat(),
                'pilot_config_sha256':sha(out/'pilot_config.json'),'train_trials_sha256':sha(out/'pilot_train_trials.jsonl'),
                'dev_trials_sha256':sha(out/'pilot_dev_trials.jsonl'),'code_sha256':code_hashes(),
                'wavlm_pretrained_sha256':sha(ROOT/'wavlm-base-plus/pytorch_model.bin')}
    checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':1,'global_step':len(logs),
                'config':cfg,'provenance':provenance,'torch_rng_state':torch.get_rng_state(),
                'python_rng_state':random.getstate(),'training_order':order,'scheduler':None}
    torch.save(checkpoint,ckpt_path)
    checkpoint_hash=sha(ckpt_path)
    del checkpoint,optimizer
    dev=score(model,out,readl(out/'pilot_dev_trials.jsonl'),checkpoint_hash,'pilot_dev_scores.jsonl')
    trials=readl(out/'singfake_trials.jsonl')
    measured=score(model,out,trials,checkpoint_hash,'singfake_scores.jsonl')
    metrics={'status':'exploratory_pilot_metrics_not_comparable_to_paper','checkpoint_sha256':checkpoint_hash,
             'protocol':'one center 4-second window per available file; T03 has two derived variants per T02 parent',
             'dev':evaluate_all([r['score'] for r in dev],[r['label_id'] for r in dev]),
             'singfake_by_subset':{},'singfake_by_codec':{},
             'singfake_overall':evaluate_all([r['score'] for r in measured],[r['label_id'] for r in measured])}
    for subset in ['T01','T02','T03']:
        rows=[r for r in measured if r['subset']==subset]
        metrics['singfake_by_subset'][subset]=evaluate_all([r['score'] for r in rows],[r['label_id'] for r in rows])
    for codec in CODECS:
        rows=[r for r in measured if r['codec_id']==codec['id']]
        metrics['singfake_by_codec'][codec['id']]=evaluate_all([r['score'] for r in rows],[r['label_id'] for r in rows])
    save(out/'pilot_metrics.json',metrics)
    save(out/'checkpoint_manifest.json',dict(provenance,checkpoint_sha256=checkpoint_hash,size_bytes=ckpt_path.stat().st_size))
    print('checkpoint and all trial scores saved',flush=True)


def code_hashes():
    paths=sorted((CODE/'opera').rglob('*.py'))+[Path(__file__).resolve()]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}


def verify(out):
    torch.set_num_threads(4)
    trials=readl(out/'singfake_trials.jsonl')
    scores=readl(out/'singfake_scores.jsonl')
    assert len(trials)==len(scores)==681
    assert [r['trial_id'] for r in trials]==[r['trial_id'] for r in scores]
    ckpt_hash=sha(out/'pilot_not_paper.pth')
    for r,s in zip(trials,scores):
        assert sha(out/r['audio_path'])==r['output_sha256']==s['audio_sha256']
        assert r['label_id']==s['label_id'] and s['checkpoint_sha256']==ckpt_hash
        if r['parent_trial_id']:
            assert r['parent_trial_id'] in {t['trial_id'] for t in trials}
    ckpt=torch.load(out/'pilot_not_paper.pth',map_location='cpu',weights_only=True)
    assert ckpt['provenance']['status']==STATUS and ckpt['global_step']==16
    assert ckpt['provenance']['code_sha256']==code_hashes()
    model=build_model('opera_net',ckpt['config']['model']['params'])
    model.load_state_dict(ckpt['model'],strict=True)
    model.eval()
    optimizer=torch.optim.AdamW(model.param_groups(1e-4,1e-5))
    optimizer.load_state_dict(ckpt['optimizer'])
    replay=[]
    with torch.inference_mode():
        for offset in [0,88,284,480,680]:
            # Replay identical batch boundaries to remove batch-size-dependent roundoff.
            start=(offset//4)*4
            rows=trials[start:start+4]
            x,_=load_batch(out,rows)
            logits=model(x)
            actual=(logits[:,1]-logits[:,0]).numpy()
            expected=np.array([r['score'] for r in scores[start:start+4]])
            assert np.allclose(actual,expected,rtol=1e-5,atol=1e-5)
            replay.append(dict(start=start,count=len(rows),max_abs_error=float(np.max(np.abs(actual-expected)))))
    save(out/'verification.json',{'status':'passed','artifact_status':STATUS,'trial_count':len(trials),
                                 'score_count':len(scores),'checkpoint_strict_reload':True,'optimizer_reload':True,
                                 'all_audio_hashes_verified':True,'score_replay':replay,
                                 'checkpoint_sha256':ckpt_hash,'not_verified':'paper performance and author-original provenance'})
    print('verification passed',flush=True)


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('phase',choices=['prepare','run','verify','all'])
    ap.add_argument('--out',type=Path,default=ROOT/'restored_data/candidate_experiment_v1')
    args=ap.parse_args()
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    for name,fn in [('prepare',prepare),('run',run),('verify',verify)]:
        if args.phase in (name,'all'):
            fn(out)


if __name__=='__main__':
    main()
