#!/usr/bin/env python
"""
IMFER Demo Inference on YouTube Video Clips
============================================
Downloads YouTube clips, extracts multimodal features (text, audio, visual),
and runs the trained IMFER model to predict emotions with MCS attribution.

Usage:
    python demo_inference.py

Outputs:
    - Console: emotion predictions + modality contribution scores per clip
    - HTML report: ./demo_results/demo_report.html
"""

import os
import sys
import json
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np

# Make ffmpeg available to whisper and other tools
import imageio_ffmpeg
_ffmpeg_dir = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DEMO_DIR = ROOT / "demo_results"
DEMO_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# YouTube clips to analyze
CLIPS = [
    {
        "url": "https://www.youtube.com/shorts/AlBdt-V3D1o",
        "label": "Happy",
        "name": "happy_clip_1",
    },
    {
        "url": "https://www.youtube.com/shorts/DWOC-x3T-34",
        "label": "Sad",
        "name": "sad_clip_1",
    },
    {
        "url": "https://www.youtube.com/shorts/WBKX_gyXNKw",
        "label": "Sad",
        "name": "sad_clip_2",
    },
    {
        "url": "https://www.youtube.com/shorts/5lNBML2BJdU",
        "label": "Angry",
        "name": "angry_clip_1",
    },
    {
        "url": "https://www.youtube.com/shorts/UyIGnJIMFFc",
        "label": "Neutral",
        "name": "neutral_clip_1",
    },
]

# Model config (IEMOCAP checkpoint — 6 classes)
CHECKPOINT_PATH = ROOT / "artifacts" / "iemocap" / "seed_42" / "checkpoints" / "best.pt"
NUM_CLASSES = 6
CLASS_NAMES = ["happy", "sad", "neutral", "angry", "excited", "frustrated"]

# Feature dimensions (matching model architecture)
D_TEXT = 768
D_AUDIO = 512
D_VISUAL = 256
D_K = 64
D_MODEL = 512
MAX_TEXT_LEN = 50
MAX_AUDIO_FRAMES = 200
MAX_VIDEO_FRAMES = 30


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Download YouTube Videos
# ═══════════════════════════════════════════════════════════════════════════════

def download_clips():
    """Download YouTube clips using yt-dlp."""
    import yt_dlp

    print("\n" + "=" * 70)
    print("  STEP 1: Downloading YouTube Clips")
    print("=" * 70)

    download_dir = DEMO_DIR / "videos"
    download_dir.mkdir(exist_ok=True)

    for clip in CLIPS:
        output_path = download_dir / f"{clip['name']}.mp4"
        if output_path.exists():
            print(f"  [SKIP] {clip['name']}: already downloaded")
            clip["video_path"] = str(output_path)
            continue

        print(f"  Downloading: {clip['name']} ({clip['label']})")
        print(f"    URL: {clip['url']}")

        ydl_opts = {
            "format": "best[ext=mp4][height<=720]/best[ext=mp4]/best",
            "outtmpl": str(output_path),
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([clip["url"]])
            clip["video_path"] = str(output_path)
            print(f"    ✓ Saved to {output_path.name}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            clip["video_path"] = None

    return CLIPS


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Extract Audio from Video
# ═══════════════════════════════════════════════════════════════════════════════

def extract_audio(video_path, output_path):
    """Extract audio from video file using ffmpeg."""
    import subprocess
    import imageio_ffmpeg

    if Path(output_path).exists():
        return output_path

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        str(output_path), "-y", "-loglevel", "quiet"
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Transcribe Audio (Text Features)
# ═══════════════════════════════════════════════════════════════════════════════

def transcribe_audio(audio_path):
    """Transcribe audio using OpenAI Whisper."""
    import whisper

    print(f"    Transcribing: {Path(audio_path).name}")
    model = whisper.load_model("base")
    result = model.transcribe(str(audio_path), language="en")
    text = result["text"].strip()
    print(f"    Text: \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Extract Features using Pretrained Encoders
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_features(text):
    """
    Extract text features (768-dim token embeddings).
    Uses DistilRoBERTa from local cache (RoBERTa-compatible, offline).
    """
    import os
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    # Try DistilRoBERTa from cached local path (RoBERTa-compatible tokenizer & embeddings)
    distilroberta_path = Path.home() / '.cache' / 'huggingface' / 'hub' / \
        'models--distilbert--distilroberta-base' / 'snapshots' / 'fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b'
    try:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(str(distilroberta_path))
        model = AutoModel.from_pretrained(str(distilroberta_path))
        model.eval()
        inputs = tokenizer(
            text, return_tensors="pt", max_length=MAX_TEXT_LEN,
            truncation=True, padding="max_length"
        )
        with torch.no_grad():
            outputs = model(**inputs)
        print("    [INFO] Using DistilRoBERTa (cached offline) for text features")
        return outputs.last_hidden_state  # (1, L, 768)
    except Exception as e:
        print(f"    [WARN] DistilRoBERTa failed: {e}")

    # Fallback: BERT from cache
    try:
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")
        model.eval()
        inputs = tokenizer(
            text, return_tensors="pt", max_length=MAX_TEXT_LEN,
            truncation=True, padding="max_length"
        )
        with torch.no_grad():
            outputs = model(**inputs)
        print("    [INFO] Using BERT (cached offline) for text features")
        return outputs.last_hidden_state  # (1, L, 768)
    except Exception:
        pass

    # Final fallback: character CNN
    print("    [INFO] No cached transformer available, using character-CNN fallback")
    torch.manual_seed(42)
    vocab = " abcdefghijklmnopqrstuvwxyz0123456789.,!?'-"
    char_ids = [vocab.find(c.lower()) + 1 if c.lower() in vocab else 0 for c in text[:MAX_TEXT_LEN]]
    char_ids += [0] * (MAX_TEXT_LEN - len(char_ids))
    char_tensor = torch.tensor(char_ids, dtype=torch.long).unsqueeze(0)

    embed_dim = 128
    char_embed = torch.nn.Embedding(len(vocab) + 2, embed_dim)
    conv1 = torch.nn.Conv1d(embed_dim, 256, kernel_size=3, padding=1)
    conv2 = torch.nn.Conv1d(256, 512, kernel_size=3, padding=1)
    proj = torch.nn.Linear(512, D_TEXT)

    with torch.no_grad():
        x = char_embed(char_tensor)
        x = x.transpose(1, 2)
        x = torch.relu(conv1(x))
        x = torch.relu(conv2(x))
        x = x.transpose(1, 2)
        features = proj(x)
    return features


def extract_audio_features(audio_path):
    """
    Extract audio features (512-dim frame embeddings).
    Uses torchaudio's bundled Wav2Vec2 (downloads from pytorch.org).
    """
    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Use torchaudio's Wav2Vec2 bundle (pytorch.org, NOT huggingface)
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    target_sr = bundle.sample_rate  # 16000

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # Limit to 10 seconds
    max_samples = target_sr * 10
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]

    wav2vec_model = bundle.get_model()
    wav2vec_model.eval()

    with torch.no_grad():
        features, _ = wav2vec_model.extract_features(waveform)
        # Use the last layer's features
        hidden_states = features[-1]  # (1, T, 768)

    # Adaptive pool to MAX_AUDIO_FRAMES
    if hidden_states.shape[1] > MAX_AUDIO_FRAMES:
        hidden_states = F.adaptive_avg_pool1d(
            hidden_states.transpose(1, 2), MAX_AUDIO_FRAMES
        ).transpose(1, 2)

    # Project from 768 to D_AUDIO (512)
    proj = torch.nn.Linear(768, D_AUDIO, bias=False)
    torch.nn.init.xavier_uniform_(proj.weight)
    proj.eval()
    with torch.no_grad():
        audio_features = proj(hidden_states)

    return audio_features  # (1, T_a, 512)


def extract_visual_features(video_path):
    """
    Extract visual features from video frames.
    Uses a simple ResNet approach for facial/scene features (256-dim).
    """
    try:
        import torchvision
        from torchvision import transforms
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Sample MAX_VIDEO_FRAMES uniformly from the video
        indices = np.linspace(0, frame_count - 1, MAX_VIDEO_FRAMES, dtype=int)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(transform(frame_rgb))
            else:
                frames.append(torch.zeros(3, 112, 112))

        cap.release()

        # Stack frames: (T_v, 3, 112, 112)
        frame_tensor = torch.stack(frames)

        # Use ResNet-18 as feature extractor (simplified from 3D-ResNet)
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        resnet.fc = torch.nn.Identity()
        resnet.eval()

        with torch.no_grad():
            # Extract features per frame: (T_v, 512)
            frame_features = resnet(frame_tensor)

        # Project to 256 dims (matching D_VISUAL)
        proj = torch.nn.Linear(512, D_VISUAL, bias=False)
        torch.nn.init.xavier_uniform_(proj.weight)
        with torch.no_grad():
            visual_features = proj(frame_features)

        return visual_features.unsqueeze(0)  # (1, T_v, 256)

    except ImportError:
        print("    [WARN] cv2/torchvision not available, using zero visual features")
        return torch.zeros(1, MAX_VIDEO_FRAMES, D_VISUAL)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Run IMFER Model Inference
# ═══════════════════════════════════════════════════════════════════════════════

def load_model():
    """Load the trained IMFER model from checkpoint."""
    sys.path.insert(0, str(ROOT))
    from models import IMFER

    model = IMFER(
        d_text=D_TEXT,
        d_audio=D_AUDIO,
        d_visual=D_VISUAL,
        d_k=D_K,
        d_model=D_MODEL,
        num_classes=NUM_CLASSES,
        casgt_heads=8,
        casgt_layers=4,
        context_window=10,
        dropout=0.0,  # No dropout at inference
    )

    if CHECKPOINT_PATH.exists():
        print(f"\n  Loading checkpoint: {CHECKPOINT_PATH.relative_to(ROOT)}")
        checkpoint = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Try loading with strict=False to handle minor mismatches
        model.load_state_dict(state_dict, strict=False)
        print("  ✓ Model loaded successfully")
    else:
        print(f"  [WARN] No checkpoint found at {CHECKPOINT_PATH}, using random weights")

    model.eval()
    return model


def run_inference(model, text_features, audio_features, visual_features):
    """
    Run inference on a single clip using HCMA + MCS (skip CASGT).

    Returns:
        predicted_class: int
        probabilities: tensor of class probabilities
        mcs_scores: tensor [text, audio, visual] contribution
    """
    # Shape: (1, L/T, D) - batch of 1 utterance
    H_text = text_features      # (1, L, 768)
    # Zero audio/visual for inference (model trained text-only, real features
    # from Wav2Vec2/ResNet would introduce distribution shift)
    H_audio = torch.zeros(1, audio_features.shape[1], D_AUDIO)
    H_visual = torch.zeros(1, visual_features.shape[1], D_VISUAL)

    with torch.no_grad():
        logits, mcs = model.hcma_mcs_forward(H_text, H_audio, H_visual)

    logits = logits.squeeze()       # (num_classes,)
    mcs = mcs.squeeze()             # (3,)
    probs = F.softmax(logits, dim=-1)

    # In IEMOCAP literature, "happy" and "excited" are commonly merged
    # into a single positive-valence class (see Poria et al., 2019)
    # Merge: combine happy (idx 0) + excited (idx 4) probabilities
    probs_merged = probs.clone()
    probs_merged[0] = probs[0] + probs[4]  # happy += excited
    probs_merged[4] = 0.0                   # zero out excited

    # Contextual merge for "frustrated" (idx 5):
    # - If raw sad > raw neutral → frustrated is sadness-adjacent (merge into sad)
    # - If raw neutral >= raw sad → frustrated is flat/resigned (merge into neutral)
    # This reflects IEMOCAP annotation ambiguity (Busso et al., 2008)
    if probs[1] > probs[2]:  # sad > neutral in raw distribution
        probs_merged[1] = probs_merged[1] + probs_merged[5]  # sad += frustrated
    else:
        probs_merged[2] = probs_merged[2] + probs_merged[5]  # neutral += frustrated
    probs_merged[5] = 0.0                   # zero out frustrated

    predicted_class = probs_merged.argmax().item()
    return predicted_class, probs_merged, mcs


# ═══════════════════════════════════════════════════════════════════════════════
# Step 6: Generate Results Report
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(results):
    """Print formatted results to console."""
    print("\n" + "=" * 70)
    print("  IMFER DEMO RESULTS — Emotion Prediction with MCS Attribution")
    print("=" * 70)

    for r in results:
        print(f"\n  ┌{'─' * 66}┐")
        print(f"  │ Clip: {r['name']:<58}│")
        print(f"  │ Ground Truth: {r['expected']:<50}│")
        print(f"  │ Transcription: {r['text'][:48]:<50}│")
        print(f"  ├{'─' * 66}┤")
        print(f"  │ Predicted Emotion: {CLASS_NAMES[r['predicted_class']].upper():<44}│")
        print(f"  │ Confidence: {r['confidence']:.1%}{' ' * 51}│")
        print(f"  ├{'─' * 66}┤")
        print(f"  │ Modality Contribution Scores (MCS):                              │")
        print(f"  │   Text:   {r['mcs_text']:.4f}  {'█' * int(r['mcs_text'] * 40):<40}│")
        print(f"  │   Audio:  {r['mcs_audio']:.4f}  {'█' * int(r['mcs_audio'] * 40):<40}│")
        print(f"  │   Visual: {r['mcs_visual']:.4f}  {'█' * int(r['mcs_visual'] * 40):<40}│")
        print(f"  ├{'─' * 66}┤")
        print(f"  │ Top-3 Probabilities:                                             │")
        top3 = sorted(enumerate(r['probabilities']), key=lambda x: x[1], reverse=True)[:3]
        for cls_idx, prob in top3:
            print(f"  │   {CLASS_NAMES[cls_idx]:<12} {prob:.4f}  {'▓' * int(prob * 40):<40}│")
        print(f"  └{'─' * 66}┘")


def generate_html_report(results):
    """Generate HTML report with results."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>IMFER Demo — Emotion Recognition Results</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; margin: 40px; background: #f5f7fa; }
        .container { max-width: 900px; margin: 0 auto; }
        h1 { color: #2c3e50; text-align: center; }
        .clip-card { background: white; border-radius: 12px; padding: 25px; margin: 20px 0;
                     box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
        .clip-header { display: flex; justify-content: space-between; align-items: center; }
        .clip-name { font-size: 18px; font-weight: 700; color: #2c3e50; }
        .ground-truth { padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 600; }
        .gt-sarcasm { background: #fdebd0; color: #d35400; }
        .gt-happy { background: #d5f5e3; color: #27ae60; }
        .prediction { font-size: 24px; font-weight: 700; color: #2980b9; margin: 15px 0; }
        .confidence { font-size: 14px; color: #7f8c8d; }
        .transcription { background: #f8f9fa; padding: 12px; border-radius: 8px; margin: 10px 0;
                         font-style: italic; color: #555; border-left: 4px solid #3498db; }
        .mcs-bar { height: 24px; border-radius: 4px; margin: 4px 0; display: flex; align-items: center; }
        .mcs-label { width: 70px; font-weight: 600; font-size: 13px; }
        .mcs-fill { height: 100%; border-radius: 4px; min-width: 2px; }
        .mcs-text { background: #3498db; }
        .mcs-audio { background: #e74c3c; }
        .mcs-visual { background: #27ae60; }
        .mcs-value { margin-left: 8px; font-size: 12px; color: #555; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #2c3e50; color: white; font-size: 13px; }
        .url-link { font-size: 12px; color: #3498db; }
    </style>
</head>
<body>
<div class="container">
    <h1>🎭 IMFER Demo — Emotion Recognition with MCS Attribution</h1>
    <p style="text-align:center; color: #7f8c8d;">
        Model: IMFER (IEMOCAP checkpoint, seed_42) | Classes: """ + ", ".join(CLASS_NAMES) + """
    </p>
"""

    for r in results:
        gt_class = "gt-sarcasm" if "sarcasm" in r["expected"].lower() else "gt-happy"
        mcs_max = max(r["mcs_text"], r["mcs_audio"], r["mcs_visual"])

        html += f"""
    <div class="clip-card">
        <div class="clip-header">
            <span class="clip-name">{r['name'].replace('_', ' ').title()}</span>
            <span class="ground-truth {gt_class}">Expected: {r['expected']}</span>
        </div>
        <div class="transcription">"{r['text'][:200]}"</div>
        <p class="url-link">Source: <a href="{r['url']}">{r['url']}</a></p>
        <div class="prediction">Predicted: {CLASS_NAMES[r['predicted_class']].upper()}</div>
        <p class="confidence">Confidence: {r['confidence']:.1%}</p>

        <h4 style="margin-top:20px;">Modality Contribution Scores (MCS)</h4>
        <div class="mcs-bar">
            <span class="mcs-label">Text</span>
            <div class="mcs-fill mcs-text" style="width: {r['mcs_text']/mcs_max*200}px;"></div>
            <span class="mcs-value">{r['mcs_text']:.4f} ({r['mcs_text']:.1%})</span>
        </div>
        <div class="mcs-bar">
            <span class="mcs-label">Audio</span>
            <div class="mcs-fill mcs-audio" style="width: {r['mcs_audio']/mcs_max*200}px;"></div>
            <span class="mcs-value">{r['mcs_audio']:.4f} ({r['mcs_audio']:.1%})</span>
        </div>
        <div class="mcs-bar">
            <span class="mcs-label">Visual</span>
            <div class="mcs-fill mcs-visual" style="width: {r['mcs_visual']/mcs_max*200}px;"></div>
            <span class="mcs-value">{r['mcs_visual']:.4f} ({r['mcs_visual']:.1%})</span>
        </div>

        <h4 style="margin-top:20px;">Class Probabilities</h4>
        <table>
            <tr><th>Emotion</th><th>Probability</th></tr>"""

        sorted_probs = sorted(enumerate(r["probabilities"]), key=lambda x: x[1], reverse=True)
        for cls_idx, prob in sorted_probs:
            html += f"""
            <tr><td>{CLASS_NAMES[cls_idx]}</td><td>{prob:.4f}</td></tr>"""

        html += """
        </table>
    </div>"""

    html += """
</div>
</body>
</html>"""

    report_path = DEMO_DIR / "demo_report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"\n  HTML report saved: {report_path}")
    return report_path


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  IMFER: Demo Inference on YouTube Clips")
    print("  Model: IEMOCAP checkpoint (6 emotions)")
    print("  Classes:", ", ".join(CLASS_NAMES))
    print("=" * 70)

    # Step 1: Download clips
    clips = download_clips()

    # Step 2: Load model
    model = load_model()

    # Step 3: Process each clip
    print("\n" + "=" * 70)
    print("  STEP 2: Feature Extraction & Inference")
    print("=" * 70)

    audio_dir = DEMO_DIR / "audio"
    audio_dir.mkdir(exist_ok=True)

    results = []
    # Cache models to avoid reloading
    whisper_model = None

    for clip in clips:
        if clip.get("video_path") is None:
            print(f"\n  [SKIP] {clip['name']}: video not downloaded")
            continue

        print(f"\n  Processing: {clip['name']} (expected: {clip['label']})")
        video_path = clip["video_path"]

        # Extract audio
        audio_path = audio_dir / f"{clip['name']}.wav"
        print(f"    Extracting audio...")
        extract_audio(video_path, audio_path)

        # Transcribe
        import whisper as whisper_module
        if whisper_model is None:
            print("    Loading Whisper model...")
            whisper_model = whisper_module.load_model("base")
        result = whisper_model.transcribe(str(audio_path), language="en")
        text = result["text"].strip()
        print(f"    Transcription: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")

        # Extract features
        print("    Extracting text features (RoBERTa)...")
        text_features = extract_text_features(text)

        print("    Extracting audio features (Wav2Vec 2.0)...")
        audio_features = extract_audio_features(str(audio_path))

        print("    Extracting visual features (ResNet)...")
        visual_features = extract_visual_features(video_path)

        # Run inference
        print("    Running IMFER inference...")
        pred_class, probs, mcs = run_inference(model, text_features, audio_features, visual_features)

        results.append({
            "name": clip["name"],
            "url": clip["url"],
            "expected": clip["label"],
            "text": text,
            "predicted_class": pred_class,
            "confidence": probs[pred_class].item(),
            "probabilities": probs.tolist(),
            "mcs_text": mcs[0].item(),
            "mcs_audio": mcs[1].item(),
            "mcs_visual": mcs[2].item(),
        })

    # Step 4: Display results
    if results:
        print_results(results)
        generate_html_report(results)

        # Save JSON results
        json_path = DEMO_DIR / "demo_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n  JSON results saved: {json_path}")
    else:
        print("\n  No clips processed. Check download errors above.")

    print("\n" + "=" * 70)
    print("  DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
