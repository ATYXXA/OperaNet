"""Verify current PDF settings. Model checks really run; failures never become PASS."""
import argparse
import json
import sys
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument("--config",type=Path,default=ROOT/"configs/opera_net.yaml")
    ap.add_argument("--json_out",type=Path,default=ROOT.parent/"restored_data/current_pdf/model_validation.json")
    ap.add_argument("--model",action="store_true",help="Run local WavLM + nnAudio forward/backward on two real training examples")
    args=ap.parse_args()
    cfg=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ref=json.loads((ROOT/"docs/paper_reference.json").read_text(encoding="utf-8"))["explicit_parameters"]
    p=cfg["model"]["params"]
    for key in ["sample_rate","fmin","n_bins","bins_per_octave","hop_length"]:
        assert p[key]==ref[key],key
    assert p["n_frozen_layers"]==6
    assert cfg["data"]["train"]["splits"]==["train"]
    assert cfg["data"]["eval"]["splits"]==["dev"]
    wavcfg=json.loads((ROOT.parent/"wavlm-base-plus/config.json").read_text())
    frames=64000
    for kernel,stride in zip(wavcfg["conv_kernel"],wavcfg["conv_stride"]):
        frames=(frames-kernel)//stride+1
    result={"explicit_parameters_checked":True,"wavlm_frames_from_local_conv_config":frames,
            "model_check":"not_requested","assumptions":{"cqt_crop_frames":p["n_frames"],"aligned_frames":p["target_frames"],"feature_dim":p["feat_dim"]}}
    if args.model:
        import torch
        import soundfile as sf
        from opera.audio import fix_length
        from opera.models import build_model
        from opera.cqt import NnAudioCQT
        torch.set_num_threads(4)
        torch.manual_seed(42)
        data=ROOT.parent/"CtrSVDD2024_Baseline/dataset"
        picked={}
        for line in (data/"train.txt").read_text().splitlines():
            fields=line.split()
            label=int(fields[-1]!="bonafide")
            picked.setdefault(label,fields[2])
            if len(picked)==2:
                break
        waves=[]
        for label in [0,1]:
            y,sr=sf.read(str(data/"train_set"/(picked[label]+".flac")),dtype="float32")
            assert sr==16000
            waves.append(fix_length(torch.from_numpy(y),64000,"center"))
        wav=torch.stack(waves)
        model=build_model(cfg["model"]["name"],p)
        assert isinstance(model.frontend.cqt,NnAudioCQT)
        groups=model.param_groups(1e-4,1e-5)
        used=[id(x) for g in groups for x in g["params"]]
        trainable={id(x) for x in model.parameters() if x.requires_grad}
        assert len(used)==len(set(used)) and set(used)==trainable,"optimizer coverage"
        model.train()
        assert all(not any(q.requires_grad for q in l.parameters()) for l in model.semantic_stream.model.encoder.layers[:6])
        assert all(any(q.requires_grad for q in l.parameters()) for l in model.semantic_stream.model.encoder.layers[6:])
        with torch.no_grad():
            raw=model.frontend.cqt(wav)
            pc=model.frontend(wav)
            f_sig=model.encoder(pc)
            native=model.semantic_stream.model(wav).last_hidden_state
            f_sem=model.semantic_stream(wav)
        assert native.shape[1]==frames
        assert f_sig.shape==f_sem.shape
        assert pc.shape==(2,2,84,200)
        logits,gate=model(wav,return_gate=True)
        assert logits.shape==(2,2)
        assert gate.shape==f_sig.shape and torch.all((gate>=0)&(gate<=1))
        loss=torch.nn.functional.cross_entropy(logits,torch.tensor([0,1]))
        loss.backward()
        grads=[q.grad for q in model.parameters() if q.requires_grad and q.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert model.semantic_stream.proj.weight.grad.abs().sum()>0
        before=model.semantic_stream.proj.weight.detach().clone()
        optim=torch.optim.AdamW(groups)
        optim.step()
        assert not torch.equal(before,model.semantic_stream.proj.weight)
        result.update(model_check="passed_forward_backward_optimizer_step",audio_ids=[picked[0],picked[1]],
                      torch_version=torch.__version__,device="cpu",raw_cqt_shape=list(raw.shape),
                      pccqt_shape=list(pc.shape),wavlm_native_shape=list(native.shape),
                      signal_shape=list(f_sig.shape),semantic_shape=list(f_sem.shape),
                      logits_shape=list(logits.shape),gate_shape=list(gate.shape),
                      optimizer_covers_all_trainable_parameters=True,
                      note="Smoke check with newly initialized detector layers. No trained detector checkpoint or paper metric is produced.")
    args.json_out.parent.mkdir(parents=True,exist_ok=True)
    args.json_out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
