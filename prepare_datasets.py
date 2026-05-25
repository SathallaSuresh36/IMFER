import argparse
import json
from pathlib import Path


DATASETS = ("iemocap", "meld", "emorynlp")
SPLIT_FILES = ("train_align.pkl", "valid_align.pkl", "test_align.pkl")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def dataset_dir(datasets_dir: Path, dataset: str) -> Path:
    if dataset == "iemocap":
        return datasets_dir / "IEMOCAP"
    if dataset == "meld":
        return datasets_dir / "MELD"
    return datasets_dir / "EmoryNLP"


def run_generator_if_needed(root: Path, dataset: str, datasets_dir: Path):
    actions = []
    if dataset == "meld":
        meld_dir = dataset_dir(datasets_dir, "meld")
        metadata = meld_dir / "metadata.csv"
        source_ann = meld_dir / "source_annotations"
        if (not metadata.exists()) and source_ann.exists():
            import subprocess

            subprocess.run([
                "d:/sureshsathalla/.venv/Scripts/python.exe",
                str(root / "generate_meld_metadata.py"),
            ], check=True, cwd=root)
            actions.append("generated meld metadata.csv")

    if dataset == "emorynlp":
        emory_dir = dataset_dir(datasets_dir, "emorynlp")
        metadata = emory_dir / "metadata.csv"
        emory_json = emory_dir / "json"
        if (not metadata.exists()) and emory_json.exists():
            import subprocess

            subprocess.run([
                "d:/sureshsathalla/.venv/Scripts/python.exe",
                str(root / "generate_emorynlp_metadata.py"),
            ], check=True, cwd=root)
            actions.append("generated emorynlp metadata.csv")

        split_paths = [emory_dir / x for x in SPLIT_FILES]
        if not all(p.exists() for p in split_paths):
            import subprocess

            subprocess.run([
                "d:/sureshsathalla/.venv/Scripts/python.exe",
                str(root / "generate_emorynlp_align.py"),
                "--out_dirs",
                "datasets/EmoryNLP",
            ], check=True, cwd=root)
            actions.append("generated emorynlp align pkl files")

    return actions


def summarize_dataset(datasets_dir: Path, dataset: str):
    d_repo = dataset_dir(datasets_dir, dataset)
    metadata = d_repo / "metadata.csv"
    align_presence = {
        name: (d_repo / name).exists() for name in SPLIT_FILES
    }

    return {
        "dataset": dataset,
        "metadata_exists": metadata.exists(),
        "align_files_present": align_presence,
        "datasets_dir": str(d_repo),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare and sync dataset build setup for IEMOCAP, MELD, and EmoryNLP.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--report", type=str, default="./artifacts/dataset_setup_report.json")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    datasets_dir = root / "datasets"

    ensure_dir(datasets_dir)

    report = {"root": str(root), "datasets": [], "actions": []}

    for dataset in DATASETS:
        gen_actions = run_generator_if_needed(root, dataset, datasets_dir)
        report["actions"].extend([f"{dataset}: {a}" for a in gen_actions])

        report["datasets"].append(summarize_dataset(datasets_dir, dataset))

    report_path = Path(args.report)
    ensure_dir(report_path.parent)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Dataset setup report written:", report_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
