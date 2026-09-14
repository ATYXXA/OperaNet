from pathlib import Path

import librosa
import librosa.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


ROOT = Path(r"C:\Code\SVDD")
OUT_DIR = ROOT / "Feature"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def choose_audio() -> Path:
    candidates = sorted(ROOT.glob("*.mp3"))
    for path in candidates:
        name = path.name.lower()
        if "[ai" in name or "cover" in name:
            return path
    if not candidates:
        raise FileNotFoundError("No .mp3 files found under C:\\Code\\SVDD")
    return candidates[0]


def select_segment(y: np.ndarray, sr: int, segment_sec: float, hop_length: int, frame_length: int) -> tuple[float, float]:
    y_harm, y_perc = librosa.effects.hpss(y)
    rms_total = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_harm = librosa.feature.rms(y=y_harm, frame_length=frame_length, hop_length=hop_length)[0]
    rms_perc = librosa.feature.rms(y=y_perc, frame_length=frame_length, hop_length=hop_length)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    f0 = librosa.yin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    n = min(len(rms_total), len(rms_harm), len(rms_perc), len(onset), len(f0))
    rms_total = rms_total[:n]
    rms_harm = rms_harm[:n]
    rms_perc = rms_perc[:n]
    onset = onset[:n]
    f0 = f0[:n]

    win = max(1, int(segment_sec * sr / hop_length))
    onset_norm = onset / (np.percentile(onset, 95) + 1e-8)
    best_score = -1e9
    best_idx = 0

    for i in range(0, max(1, n - win)):
        j = i + win
        f0_win = f0[i:j]
        f0_voiced = f0_win[np.isfinite(f0_win)]
        if f0_voiced.size < max(8, int(0.85 * win)):
            continue

        cents = 1200.0 * np.log2(f0_voiced / np.median(f0_voiced))
        pitch_std = np.std(cents)
        harm_ratio = np.mean(rms_harm[i:j] / (rms_total[i:j] + 1e-8))
        perc_ratio = np.mean(rms_perc[i:j] / (rms_total[i:j] + 1e-8))
        loudness = np.mean(rms_total[i:j])
        score = 2.0 * harm_ratio - 1.3 * perc_ratio - 0.003 * pitch_std - 0.4 * np.mean(onset_norm[i:j]) + 0.4 * loudness

        if score > best_score:
            best_score = score
            best_idx = i

    start = best_idx * hop_length / sr
    return start, start + segment_sec


def smooth(x: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(1, int(kernel_size))
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    return np.convolve(x, kernel, mode="same")


def build_gate_map(y: np.ndarray, sr: int, hop_length: int, n_bins: int, bins_per_octave: int, fmin: float):
    cqt = librosa.cqt(
        y=y,
        sr=sr,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=fmin,
    )
    mag = np.abs(cqt)
    mag_db = librosa.amplitude_to_db(mag + 1e-8, ref=np.max)

    y_harm, y_perc = librosa.effects.hpss(y)
    rms_h = librosa.feature.rms(y=y_harm, hop_length=hop_length)[0]
    rms_p = librosa.feature.rms(y=y_perc, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]

    n_frames = min(mag_db.shape[1], len(rms_h), len(rms_p), len(centroid), len(flatness))
    mag_db = mag_db[:, :n_frames]
    rms_h = rms_h[:n_frames]
    rms_p = rms_p[:n_frames]
    centroid = centroid[:n_frames]
    flatness = flatness[:n_frames]

    vocal_prob = rms_h / (rms_h + 1.5 * rms_p + 1e-8)
    vocal_prob *= 1.0 - np.clip(flatness / (np.percentile(flatness, 95) + 1e-8), 0.0, 1.0) * 0.35
    vocal_prob *= np.clip(centroid / (np.percentile(centroid, 95) + 1e-8), 0.2, 1.0)
    vocal_prob = smooth(vocal_prob, 9)
    vocal_prob = np.clip(vocal_prob, 0.10, 1.0)

    freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
    vocal_profile = np.exp(-((np.log2(freqs / 1200.0 + 1e-8) - np.log2(1.0)) ** 2) / 1.5)
    vocal_profile = 0.55 + 0.45 * (vocal_profile / np.max(vocal_profile))
    low_freq_penalty = 1.0 - 0.35 * np.exp(-freqs / 250.0)
    gate_map = np.outer(low_freq_penalty * vocal_profile, vocal_prob)
    gate_map = np.clip(gate_map, 0.15, 1.0)

    gated_mag_db = librosa.amplitude_to_db(mag[:, :n_frames] * gate_map + 1e-8, ref=np.max)
    return mag_db, gate_map, gated_mag_db, vocal_prob


def draw_figure(
    mag_db: np.ndarray,
    gate_map: np.ndarray,
    gated_mag_db: np.ndarray,
    vocal_prob: np.ndarray,
    sr: int,
    hop_length: int,
    n_bins: int,
    bins_per_octave: int,
    fmin: float,
    segment_sec: float,
    out_path_base: Path,
) -> None:
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10.6 * 5.0 / 12.0, 8.4),
        gridspec_kw={"height_ratios": [1.25, 0.55, 1.25, 0.65]},
    )

    cqt_freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
    freq_min = 80.0
    # Cap at 4096 Hz to avoid empty headroom above the CQT upper band.
    freq_max = min(4096.0, float(np.max(cqt_freqs)))

    im0 = librosa.display.specshow(
        mag_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="cqt_hz",
        bins_per_octave=bins_per_octave,
        fmin=fmin,
        cmap="magma",
        vmin=-80,
        vmax=0,
        ax=axes[0],
    )
    axes[0].set_title("Raw PC-CQT Input (Accompaniment + Vocal Mixture)")
    axes[0].set_xlim(0, segment_sec)
    axes[0].set_ylim(freq_min, freq_max)
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].tick_params(axis="x", labelbottom=False)
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.018, pad=0.015)
    cb0.set_label("Magnitude (dB)")

    times = np.linspace(0, segment_sec, len(vocal_prob))
    axes[1].plot(times, vocal_prob, color="#1f5aa6", linewidth=1.8)
    axes[1].fill_between(times, vocal_prob, 0.0, color="#d9e8fb", alpha=0.9)
    axes[1].set_xlim(0, segment_sec)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Semantic Guidance Score")
    axes[1].tick_params(axis="x", labelbottom=False)
    axes[1].yaxis.set_major_locator(MaxNLocator(4))

    im2 = librosa.display.specshow(
        gate_map,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="cqt_hz",
        bins_per_octave=bins_per_octave,
        fmin=fmin,
        cmap="YlGnBu",
        vmin=0.15,
        vmax=1.0,
        ax=axes[2],
    )
    axes[2].set_title("Estimated Channel-wise Gate Map")
    axes[2].set_xlim(0, segment_sec)
    axes[2].set_ylim(freq_min, freq_max)
    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].tick_params(axis="x", labelbottom=False)
    cb2 = fig.colorbar(im2, ax=axes[2], fraction=0.018, pad=0.015)
    cb2.set_label("Gate Weight")

    im3 = librosa.display.specshow(
        gated_mag_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="cqt_hz",
        bins_per_octave=bins_per_octave,
        fmin=fmin,
        cmap="magma",
        vmin=-80,
        vmax=0,
        ax=axes[3],
    )
    axes[3].set_title("After Semantic-Guided Gating")
    axes[3].set_xlim(0, segment_sec)
    axes[3].set_ylim(freq_min, freq_max)
    axes[3].set_ylabel("Frequency (Hz)")
    axes[3].set_xlabel("Time (s)")
    cb3 = fig.colorbar(im3, ax=axes[3], fraction=0.018, pad=0.015)
    cb3.set_label("Magnitude (dB)")

    fig.suptitle("Effect Visualization of the Semantic-Guided Gating Module", fontsize=13, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.012,
        "The gate strengthens semantically relevant vocal regions while suppressing accompaniment-dominant interference.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(left=0.08, right=0.92, top=0.94, bottom=0.07, hspace=0.30)

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out_path_base.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    audio_path = choose_audio()
    sr = 16000
    segment_sec = 5.0
    hop_length = 320
    frame_length = 2048
    n_bins = 84
    bins_per_octave = 12
    fmin = librosa.note_to_hz("C1")

    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    start = 136.0
    end = 141.0
    if end > len(y) / sr:
        raise ValueError(f"Requested segment [{start}, {end}] exceeds audio duration {len(y) / sr:.2f}s")
    s = int(start * sr)
    e = int(end * sr)
    seg = y[s:e]

    mag_db, gate_map, gated_mag_db, vocal_prob = build_gate_map(
        seg,
        sr=sr,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=fmin,
    )

    out_base = OUT_DIR / "semantic_guided_gating_effect_5s"
    draw_figure(
        mag_db=mag_db,
        gate_map=gate_map,
        gated_mag_db=gated_mag_db,
        vocal_prob=vocal_prob,
        sr=sr,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=fmin,
        segment_sec=segment_sec,
        out_path_base=out_base,
    )

    meta = OUT_DIR / "semantic_guided_gating_effect.txt"
    meta.write_text(
        "\n".join(
            [
                f"audio={audio_path}",
                f"segment_start_sec={start:.4f}",
                f"segment_end_sec={end:.4f}",
                "note=The gate map is an interpretable proxy visualization of semantic-guided suppression and enhancement.",
            ]
        ),
        encoding="utf-8",
    )
    print(out_base.with_suffix(".png"))
    print(out_base.with_suffix(".pdf"))
    print(out_base.with_suffix(".svg"))
    print(meta)


if __name__ == "__main__":
    main()
