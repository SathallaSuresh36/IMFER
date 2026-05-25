import argparse
import csv
import json
from pathlib import Path


SPLITS = {
    "train": "emotion-detection-trn.json",
    "dev": "emotion-detection-dev.json",
    "test": "emotion-detection-tst.json",
}


def iter_utterances(split_name: str, json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for episode in data.get("episodes", []):
        ep_id = str(episode.get("episode_id", ""))
        for scene in episode.get("scenes", []):
            scene_id = str(scene.get("scene_id", ""))
            conv_id = f"{ep_id}_{scene_id}" if ep_id or scene_id else ""
            for idx, utt in enumerate(scene.get("utterances", [])):
                speakers = utt.get("speakers") or []
                speaker = str(speakers[0]) if speakers else "unknown"
                utt_id = str(utt.get("utterance_id", ""))
                text = str(utt.get("transcript", "")).strip()
                label = str(utt.get("emotion", "neutral")).strip().lower()

                yield {
                    "split": split_name,
                    "conversation_id": conv_id,
                    "turn_index": idx,
                    "utterance_id": utt_id,
                    "speaker_id": speaker,
                    "text": text,
                    "audio_path": "",
                    "video_path": "",
                    "label": label,
                }


def main():
    parser = argparse.ArgumentParser(description="Generate datasets/EmoryNLP/metadata.csv from official EmoryNLP JSON files.")
    parser.add_argument("--json_dir", type=Path, default=Path("datasets/EmoryNLP/json"))
    parser.add_argument("--out_csv", type=Path, default=Path("datasets/EmoryNLP/metadata.csv"))
    args = parser.parse_args()

    rows = []
    counts = {"train": 0, "dev": 0, "test": 0}

    for split, filename in SPLITS.items():
        src = args.json_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing split file: {src}")
        split_rows = list(iter_utterances(split, src))
        counts[split] = len(split_rows)
        rows.extend(split_rows)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "conversation_id",
        "turn_index",
        "utterance_id",
        "speaker_id",
        "text",
        "audio_path",
        "video_path",
        "label",
    ]

    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.out_csv} with {len(rows)} rows")
    print(f"  train: {counts['train']}")
    print(f"  dev: {counts['dev']}")
    print(f"  test: {counts['test']}")


if __name__ == "__main__":
    main()
