"""Inspect local SingFake headers and recover deterministic 4-second window indexes.

The window boundaries are a declared reconstruction, not author-provided segments.
Unknown/multiple filename-to-metadata candidates are kept in the parent inventory.
"""
import json
from collections import Counter
from pathlib import Path
import soundfile as sf
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'restored_data/current_pdf'


def main():
    rows=[json.loads(l) for l in (OUT/'singfake_audio_inventory.jsonl').read_text(encoding='utf-8').splitlines()]
    stats=Counter()
    errors=[]
    with (OUT/'singfake_audio_headers.jsonl').open('w',encoding='utf-8') as headers, (OUT/'singfake_eval_segments.jsonl').open('w',encoding='utf-8') as segments:
        for row in rows:
            try:
                info=sf.info(str(ROOT/row['path']))
                if info.duration<=0:
                    raise ValueError('empty duration')
                record=dict(file_id=row['file_id'],duration_seconds=info.duration,sample_rate=info.samplerate,
                            channels=info.channels,frames=info.frames,format=info.format,
                            verification='soundfile_header_read_not_full_decode')
                stats['header_readable']+=1
                if row['subset'] in ['T01','T02','T03']:
                    duration=info.duration
                    starts=[0.] if duration<=4 else [4.*i for i in range(int((duration-4)//4)+1)]
                    # Matches current dataset loader: final overlapping window when >=1s is left.
                    if duration>4 and duration-starts[-1]-4>=1:
                        starts.append(duration-4)
                    for start in starts:
                        segments.write(json.dumps(dict(utt_id=f"{row['file_id']}#{start:.2f}",track_id=row['file_id'],
                            path=row['path'],subset=row['subset'],label_id=row['label_id'],
                            start_seconds=start,end_seconds=min(start+4,duration),target_sample_rate=16000,
                            target_samples=64000,provenance='reconstructed_windows_from_local_audio',
                            label_provenance='local_directory_not_content_verified'),ensure_ascii=False)+'\n')
                        stats['segments_'+row['subset']]+=1
            except (RuntimeError, ValueError, OSError) as exc:
                record=dict(file_id=row['file_id'],error=str(exc))
                errors.append(record)
            headers.write(json.dumps(record,ensure_ascii=False)+'\n')
    (OUT/'singfake_audio_validation.json').write_text(json.dumps(dict(counts=dict(stats),errors=errors),ensure_ascii=False,indent=2),encoding='utf-8')
    print(dict(stats),'errors',len(errors))


if __name__=='__main__':
    main()
