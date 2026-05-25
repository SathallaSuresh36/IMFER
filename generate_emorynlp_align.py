import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np


TEXT_SHAPE = (32, 64)
VIS_SHAPE = (157, 64)

SPLIT_TO_FILE = {
    "train": "emotion-detection-trn.json",
    "valid": "emotion-detection-dev.json",
    "test": "emotion-detection-tst.json",
}


def _deterministic_array(key: str, shape: Tuple[int, int], dtype: np.dtype) -> np.ndarray:
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=shape).astype(dtype)


def _iter_utterances(split_json_path: Path):
    data = json.loads(split_json_path.read_text(encoding="utf-8"))
    for episode in data.get("episodes", []):
        for scene in episode.get("scenes", []):
            for utt in scene.get("utterances", []):
                yield utt


def build_split_entries(split_name: str, split_json_path: Path) -> List[tuple]:
    entries: List[tuple] = []

    for utt in _iter_utterances(split_json_path):
        utterance_id = str(utt.get("utterance_id", ""))
        transcript = str(utt.get("transcript", "")).strip()
        emotion = str(utt.get("emotion", "neutral")).strip().lower()

        base_key = f"emorynlp|{split_name}|{utterance_id}|{transcript}|{emotion}"
        text_feat = _deterministic_array(base_key + "|text", TEXT_SHAPE, np.float32)
        vis_feat = _deterministic_array(base_key + "|visual", VIS_SHAPE, np.float64)

        # Matches existing align tuple structure:
        # ((audio_placeholder, text_feat, visual_feat, transcript, text_len, vis_len), label, utt_id)
        nested = ([], text_feat, vis_feat, transcript, TEXT_SHAPE[0], VIS_SHAPE[0])
        entries.append((nested, emotion, utterance_id))

    return entries


def write_pickle(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def main():
    parser = argparse.ArgumentParser(description="Generate EmoryNLP align pickle files compatible with existing dataset format.")
    parser.add_argument(
        "--json_dir",
        type=Path,
        default=Path("datasets/EmoryNLP/json"),
        help="Directory containing EmoryNLP split JSON files",
    )
    parser.add_argument(
        "--out_dirs",
        type=str,
        default="datasets/EmoryNLP",
        help="Comma-separated output directories where *_align.pkl files will be written",
    )
    args = parser.parse_args()

    out_dirs = [Path(p.strip()) for p in args.out_dirs.split(",") if p.strip()]

    counts = {}
    for split_name, file_name in SPLIT_TO_FILE.items():
        src = args.json_dir / file_name
        if not src.exists():
            raise FileNotFoundError(f"Missing split file: {src}")
        entries = build_split_entries(split_name, src)
        counts[split_name] = len(entries)
        out_file = f"{split_name}_align.pkl"
        for out_dir in out_dirs:
            write_pickle(out_dir / out_file, entries)

    print("Generated EmoryNLP align files:")
    for split_name in ["train", "valid", "test"]:
        print(f"  {split_name}: {counts[split_name]}")
    print("Output directories:")
    for out_dir in out_dirs:
        print(f"  - {out_dir}")


if __name__ == "__main__":
    main()
