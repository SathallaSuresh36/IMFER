#!/usr/bin/env python
"""
IMFER: Interpretable Multimodal Fusion for Emotion Recognition
===============================================================
End-to-End Reproducibility Script — IEMOCAP · MELD · EmoryNLP

Usage:
    python IMFER_build_pipeline.py

Outputs:
    - Console: all metrics, status, and summary tables
    - HTML report: ./artifacts/reproduction_report.html
"""

from pathlib import Path
import csv
import datetime
import json
import os
import pickle
import re
import subprocess
import sys

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

DATASETS = {
    'iemocap': {
        'folder': 'IEMOCAP',
        'num_classes': 6,
        'class_names': 'happy,sad,neutral,angry,excited,frustrated',
        'metadata': ROOT / 'datasets' / 'IEMOCAP' / 'metadata.csv',
    },
    'meld': {
        'folder': 'MELD',
        'num_classes': 7,
        'class_names': 'neutral,surprise,fear,sadness,joy,disgust,anger',
        'metadata': ROOT / 'datasets' / 'MELD' / 'metadata.csv',
    },
    'emorynlp': {
        'folder': 'EmoryNLP',
        'num_classes': 7,
        'class_names': 'joyful,peaceful,powerful,scared,mad,sad,neutral',
        'metadata': ROOT / 'datasets' / 'EmoryNLP' / 'metadata.csv',
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def stage(title):
    print('\n' + '=' * 80)
    print(f'  {title}')
    print('=' * 80)


def run_cmd(args, check=True):
    print(f'\n  $ {" ".join(map(str, args))}')
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, env=env)
    if result.stdout:
        for line in result.stdout.strip().split('\n'):
            print(f'    {line}')
    if result.stderr:
        for line in result.stderr.strip().split('\n'):
            print(f'    [stderr] {line}')
    if check and result.returncode != 0:
        raise RuntimeError(f'Command failed (exit {result.returncode}): {" ".join(map(str, args))}')
    return result


def _split_name(dataset, base_split):
    if base_split == 'valid':
        return 'dev' if dataset in {'meld', 'emorynlp'} else 'val'
    return base_split


def _infer_conversation_turn(utterance_id, fallback_index):
    utt = str(utterance_id)
    if '_' in utt:
        conv = utt.rsplit('_', 1)[0]
        tail = utt.rsplit('_', 1)[1]
    else:
        conv = utt
        tail = utt
    m = re.search(r'(\d+)$', tail)
    turn_idx = int(m.group(1)) if m else fallback_index
    return conv, turn_idx


def ensure_align_and_metadata(dataset):
    cfg = DATASETS[dataset]
    datasets_dir = ROOT / 'datasets' / cfg['folder']
    split_to_file = {'train': 'train_align.pkl', 'valid': 'valid_align.pkl', 'test': 'test_align.pkl'}
    all_rows = []

    for split, file_name in split_to_file.items():
        align_path = datasets_dir / file_name
        if not align_path.exists():
            raise FileNotFoundError(f'Missing align file for {dataset} split {split}: {align_path}')
        with open(align_path, 'rb') as f:
            items = pickle.load(f)
        out_split = _split_name(dataset, split)
        for idx, item in enumerate(items):
            if not isinstance(item, tuple) or len(item) < 3:
                continue
            payload, label, utterance_id = item[0], str(item[1]).lower(), str(item[2])
            text = ''
            if isinstance(payload, tuple) and len(payload) >= 4 and isinstance(payload[3], str):
                text = payload[3].strip()
            conv_id, turn_index = _infer_conversation_turn(utterance_id, idx)
            all_rows.append({
                'split': out_split, 'conversation_id': conv_id, 'turn_index': turn_index,
                'utterance_id': utterance_id, 'speaker_id': 'unknown', 'text': text,
                'audio_path': '', 'video_path': '', 'label': label,
            })

    metadata_path = cfg['metadata']
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'split', 'conversation_id', 'turn_index', 'utterance_id',
            'speaker_id', 'text', 'audio_path', 'video_path', 'label'
        ])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'    Wrote {metadata_path} with {len(all_rows)} rows')
    return metadata_path


def find_predictions_csv(dataset):
    pred_files = sorted((ROOT / 'artifacts' / dataset).glob('seed_*/predictions/test_predictions.csv'))
    return pred_files[0] if pred_files else None


def training_args(dataset, profile):
    args = [PYTHON, 'train.py', '--dataset', dataset, '--device', 'cpu']
    if profile['seeds']:
        args += ['--seeds', profile['seeds']]
    if profile['max_epochs']:
        args += ['--max_epochs', profile['max_epochs']]
    if profile['patience']:
        args += ['--patience', profile['patience']]
    return args


# ═══════════════════════════════════════════════════════════════════════════════
# HTML Report Generator
# ═══════════════════════════════════════════════════════════════════════════════

def generate_html_report(summaries, bootstraps, class_reports, profile_name):
    datasets_list = list(DATASETS.keys())
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IMFER Reproduction Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #f5f7fa; color: #333; }}
        .container {{ max-width: 1100px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2c3e50; margin-top: 40px; }}
        h3 {{ color: #34495e; }}
        .meta {{ color: #7f8c8d; font-size: 14px; margin-bottom: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 14px; }}
        th {{ background-color: #2c3e50; color: white; padding: 12px 15px; text-align: center; }}
        td {{ padding: 10px 15px; text-align: center; border-bottom: 1px solid #ecf0f1; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        tr:hover {{ background-color: #ebf5fb; }}
        .metric-label {{ text-align: left; font-weight: 600; }}
        .highlight {{ background-color: #d5f5e3; font-weight: bold; }}
        .summary-box {{ background: #eaf2f8; padding: 20px; border-radius: 6px; margin: 20px 0; border-left: 4px solid #3498db; }}
        .figure-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }}
        .figure-card {{ border: 1px solid #ecf0f1; border-radius: 6px; padding: 15px; text-align: center; }}
        .figure-card img {{ max-width: 100%; height: auto; }}
        .figure-card p {{ font-weight: 600; color: #2c3e50; margin-top: 10px; }}
        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; }}
        .badge-success {{ background: #d5f5e3; color: #27ae60; }}
        .badge-info {{ background: #d6eaf8; color: #2980b9; }}
        footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ecf0f1; color: #95a5a6; font-size: 12px; text-align: center; }}
    </style>
</head>
<body>
<div class="container">
    <h1>IMFER: Interpretable Multimodal Fusion for Emotion Recognition</h1>
    <p class="meta">
        <span class="badge badge-info">{profile_name} Profile</span>
        &nbsp; Generated: {timestamp}
    </p>

    <div class="summary-box">
        <strong>Datasets:</strong> IEMOCAP, MELD, EmoryNLP<br>
        <strong>Evaluation:</strong> Weighted-F1, Macro-F1, Accuracy, Bootstrap 95% CI<br>
        <strong>Modalities:</strong> Text (RoBERTa), Audio (Wav2Vec2), Visual (R3D-18)
    </div>

    <h2>Table 1: Primary Metrics Comparison</h2>
    <table>
        <tr><th>Metric</th>'''

    for ds in datasets_list:
        html += f'<th>{ds.upper()}</th>'
    html += '</tr>\n'

    metrics_rows = [
        ('Weighted F1 (%)', 'wf1_mean'),
        ('Macro F1 (%)', 'mf1'),
        ('Accuracy (%)', 'accuracy'),
        ('WF1 Std Dev', 'wf1_std'),
        ('Num Seeds', 'num_runs'),
    ]

    for label, key in metrics_rows:
        html += f'    <tr><td class="metric-label">{label}</td>'
        for ds in datasets_list:
            s = summaries.get(ds, {})
            if key == 'wf1_mean':
                val = f"{s.get('wf1_mean', 0):.2f}"
            elif key == 'wf1_std':
                val = f"{s.get('wf1_std', 0):.4f}"
            elif key == 'num_runs':
                val = str(s.get('num_runs', '-'))
            elif key in ('mf1', 'accuracy'):
                runs = s.get('runs', [])
                if runs:
                    vals = [r.get(key, 0) for r in runs if r.get(key, 0) > 0]
                    val = f"{sum(vals)/len(vals):.2f}" if vals else f"{runs[0].get(key, 0):.2f}"
                else:
                    val = '-'
            else:
                val = '-'
            html += f'<td>{val}</td>'
        html += '</tr>\n'

    # Bootstrap CI row
    html += '    <tr><td class="metric-label">Bootstrap 95% CI</td>'
    for ds in datasets_list:
        b = bootstraps.get(ds, {})
        ci = b.get('bootstrap_ci95', [0, 0])
        html += f'<td>[{ci[0]:.2f}, {ci[1]:.2f}]</td>'
    html += '</tr>\n</table>\n'

    # Table 2: MCS
    html += '''
    <h2>Table 2: Modality Contribution Scores (MCS)</h2>
    <table>
        <tr><th>Modality</th>'''
    for ds in datasets_list:
        html += f'<th>{ds.upper()}</th>'
    html += '</tr>\n'

    modalities = [('Text', 'mcs_text'), ('Audio', 'mcs_audio'), ('Visual', 'mcs_visual')]
    for mod_name, mod_key in modalities:
        html += f'    <tr><td class="metric-label">{mod_name}</td>'
        for ds in datasets_list:
            s = summaries.get(ds, {})
            runs = s.get('runs', [])
            if runs and mod_key in runs[0]:
                val = f"{runs[0][mod_key]:.4f}"
            else:
                val = '-'
            html += f'<td>{val}</td>'
        html += '</tr>\n'
    html += '</table>\n'

    # Table 3: Per-class F1
    html += '''
    <h2>Table 3: Per-Class F1 Scores</h2>
    <table>
        <tr><th>Dataset</th><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
'''
    for ds in datasets_list:
        report = class_reports.get(ds, {})
        class_names = report.get('class_names', [])
        class_metrics = report.get('class_metrics', [])
        for i, (cn, cm) in enumerate(zip(class_names, class_metrics)):
            ds_cell = f'<td rowspan="{len(class_names) + 1}" class="metric-label">{ds.upper()}</td>' if i == 0 else ''
            html += f'    <tr>{ds_cell}<td>{cn}</td>'
            html += f'<td>{cm["precision"]:.4f}</td><td>{cm["recall"]:.4f}</td>'
            html += f'<td>{cm["f1"]:.4f}</td><td>{cm["support"]}</td></tr>\n'
        # Weighted avg row
        summary_data = report.get('summary', {})
        weighted = summary_data.get('weighted', {})
        overall = summary_data.get('overall', {})
        html += f'    <tr class="highlight"><td><strong>Weighted Avg</strong></td>'
        html += f'<td>{weighted.get("precision", 0):.4f}</td><td>{weighted.get("recall", 0):.4f}</td>'
        html += f'<td>{weighted.get("f1", 0):.4f}</td><td>{overall.get("support", 0)}</td></tr>\n'
    html += '</table>\n'

    # Figures section
    html += '\n    <h2>Generated Figures</h2>\n    <div class="figure-grid">\n'
    for ds in datasets_list:
        for fig_name, fig_title in [('confusion_matrix.png', 'Confusion Matrix'), ('per_class_f1.png', 'Per-Class F1')]:
            fig_path = ROOT / 'figures' / ds / fig_name
            if fig_path.exists():
                rel_path = f'../figures/{ds}/{fig_name}'
                html += f'    <div class="figure-card"><img src="{rel_path}" alt="{fig_title} - {ds}"><p>{ds.upper()} — {fig_title}</p></div>\n'
    html += '    </div>\n'

    html += f'''
    <footer>
        IMFER Reproduction Report &mdash; Generated automatically by run_pipeline.py<br>
        Profile: {profile_name} | Date: {timestamp}
    </footer>
</div>
</body>
</html>'''

    return html


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    profile = {'seeds': '42', 'max_epochs': '20', 'patience': '5'}
    profile_name = 'Quick (single seed)'

    print('\n' + '╔' + '═' * 78 + '╗')
    print('║' + ' IMFER: Interpretable Multimodal Fusion for Emotion Recognition '.center(78) + '║')
    print('║' + ' End-to-End Reproducibility Pipeline '.center(78) + '║')
    print('╚' + '═' * 78 + '╝')
    print(f'\n  Workspace:  {ROOT}')
    print(f'  Python:     {PYTHON}')
    print(f'  Profile:    {profile_name}')
    print(f'  Datasets:   {", ".join(DATASETS.keys())}')

    # ── Stage 1: Dataset Integrity Verification ──
    stage('Stage 1: Dataset Integrity Verification')
    status = {}
    for dataset, cfg in DATASETS.items():
        metadata_path = ensure_align_and_metadata(dataset)
        ds_dir = ROOT / 'datasets' / cfg['folder']
        status[f'{dataset}_metadata'] = metadata_path.exists()
        status[f'{dataset}_train_align'] = (ds_dir / 'train_align.pkl').exists()
        status[f'{dataset}_valid_align'] = (ds_dir / 'valid_align.pkl').exists()
        status[f'{dataset}_test_align'] = (ds_dir / 'test_align.pkl').exists()

    print('\n  Readiness status:')
    for k, v in status.items():
        icon = '✓' if v else '✗'
        print(f'    {icon} {k}')

    missing = [k for k, v in status.items() if not v]
    if missing:
        raise FileNotFoundError('Missing required dataset files: ' + ', '.join(missing))
    print('\n  All datasets ready.')

    # ── Stage 2: Codebase Validation ──
    stage('Stage 2: Codebase Validation')
    # Compile only project source files (skip .venv/virtual environments)
    py_files = [str(f) for f in ROOT.glob('*.py')]
    run_cmd([PYTHON, '-m', 'compileall'] + py_files)
    run_cmd([PYTHON, '-m', 'unittest', 'tests/test_data_pipeline.py', '-v'])

    # ── Stage 3: Model Training ──
    stage('Stage 3: Model Training')
    for dataset, cfg in DATASETS.items():
        pred_csv = find_predictions_csv(dataset)
        if pred_csv is not None:
            print(f'\n  [SKIP] {dataset.upper()}: Predictions exist at {pred_csv.relative_to(ROOT)}')
            continue
        print(f'\n  Training: {dataset.upper()}')
        run_cmd(training_args(dataset, profile))

    # ── Stage 4: Post-Training Evaluation ──
    stage('Stage 4: Post-Training Evaluation & Visualization')
    for dataset, cfg in DATASETS.items():
        print(f'\n  Processing: {dataset.upper()}')

        run_cmd([PYTHON, 'evaluate.py',
                 '--artifacts_root', './artifacts', '--dataset', dataset,
                 '--num_classes', str(cfg['num_classes'])])

        run_cmd([PYTHON, 'bootstrap_analysis.py',
                 '--aggregate_csv', f'./artifacts/{dataset}/aggregate/metrics.csv',
                 '--out_json', f'./artifacts/{dataset}/aggregate/bootstrap_summary.json'])

        run_cmd([PYTHON, 'visualize_results.py',
                 '--aggregate_csv', f'./artifacts/{dataset}/aggregate/metrics.csv',
                 '--output_dir', f'./figures/{dataset}'])

        pred_csv = find_predictions_csv(dataset)
        if pred_csv is None:
            raise FileNotFoundError(f'No predictions CSV for {dataset}')

        run_cmd([PYTHON, 'classification_report_and_plots.py',
                 '--predictions_csv', str(pred_csv.relative_to(ROOT)).replace('\\', '/'),
                 '--class_names', cfg['class_names'],
                 '--dataset_name', dataset,
                 '--output_dir', f'./figures/{dataset}',
                 '--report_out', f'./artifacts/{dataset}/aggregate/classification_report.txt',
                 '--json_out', f'./artifacts/{dataset}/aggregate/classification_report.json'])

    # ── Stage 5: Consolidated Results ──
    stage('Stage 5: Consolidated Results')

    summaries = {}
    bootstraps = {}
    class_reports = {}

    for dataset in DATASETS.keys():
        summary_path = ROOT / 'artifacts' / dataset / 'aggregate' / 'summary.json'
        bootstrap_path = ROOT / 'artifacts' / dataset / 'aggregate' / 'bootstrap_summary.json'
        report_path = ROOT / 'artifacts' / dataset / 'aggregate' / 'classification_report.json'

        if summary_path.exists():
            summaries[dataset] = json.loads(summary_path.read_text(encoding='utf-8'))
        if bootstrap_path.exists():
            bootstraps[dataset] = json.loads(bootstrap_path.read_text(encoding='utf-8'))
        if report_path.exists():
            class_reports[dataset] = json.loads(report_path.read_text(encoding='utf-8'))

    # Print console summary tables
    datasets_list = list(DATASETS.keys())
    col_w = 15

    print('\n  ┌─────────────────────────────────────────────────────────────────┐')
    print('  │          PRIMARY METRICS COMPARISON                             │')
    print('  ├───────────────────┬───────────────┬───────────────┬─────────────┤')
    header = '  │ {:17s} │'.format('Metric')
    for ds in datasets_list:
        header += ' {:^13s} │'.format(ds.upper())
    print(header)
    print('  ├───────────────────┼───────────────┼───────────────┼─────────────┤')

    metrics_display = [
        ('Weighted F1 (%)', lambda s: f"{s.get('wf1_mean', 0):.2f}"),
        ('Macro F1 (%)', lambda s: f"{s.get('runs', [{}])[0].get('mf1', 0):.2f}" if s.get('runs') else '-'),
        ('Accuracy (%)', lambda s: f"{s.get('runs', [{}])[0].get('accuracy', 0):.2f}" if s.get('runs') else '-'),
        ('WF1 Std Dev', lambda s: f"{s.get('wf1_std', 0):.4f}"),
        ('Num Seeds', lambda s: str(s.get('num_runs', '-'))),
        ('Bootstrap CI', lambda s: '-'),
    ]

    for label, extractor in metrics_display:
        row = '  │ {:17s} │'.format(label)
        for ds in datasets_list:
            s = summaries.get(ds, {})
            if label == 'Bootstrap CI':
                b = bootstraps.get(ds, {})
                ci = b.get('bootstrap_ci95', [0, 0])
                val = f'[{ci[0]:.1f},{ci[1]:.1f}]'
            else:
                val = extractor(s)
            row += ' {:^13s} │'.format(val)
        print(row)

    print('  └───────────────────┴───────────────┴───────────────┴─────────────┘')

    # MCS Table
    print('\n  ┌─────────────────────────────────────────────────────────────────┐')
    print('  │          MODALITY CONTRIBUTION SCORES (MCS)                     │')
    print('  ├───────────────────┬───────────────┬───────────────┬─────────────┤')
    header = '  │ {:17s} │'.format('Modality')
    for ds in datasets_list:
        header += ' {:^13s} │'.format(ds.upper())
    print(header)
    print('  ├───────────────────┼───────────────┼───────────────┼─────────────┤')

    for mod_name, mod_key in [('Text', 'mcs_text'), ('Audio', 'mcs_audio'), ('Visual', 'mcs_visual')]:
        row = '  │ {:17s} │'.format(mod_name)
        for ds in datasets_list:
            s = summaries.get(ds, {})
            runs = s.get('runs', [])
            val = f"{runs[0][mod_key]:.4f}" if runs and mod_key in runs[0] else '-'
            row += ' {:^13s} │'.format(val)
        print(row)

    print('  └───────────────────┴───────────────┴───────────────┴─────────────┘')

    # ── Stage 6: Generate HTML Report ──
    stage('Stage 6: Generate HTML Report')
    report_html = generate_html_report(summaries, bootstraps, class_reports, profile_name)
    report_path = ROOT / 'artifacts' / 'reproduction_report.html'
    report_path.write_text(report_html, encoding='utf-8')
    print(f'\n  HTML report saved to: {report_path}')

    # Final
    print('\n' + '╔' + '═' * 78 + '╗')
    print('║' + ' REPRODUCTION COMPLETE '.center(78) + '║')
    print('╚' + '═' * 78 + '╝')
    print(f'\n  Artifacts: ./artifacts/<dataset>/aggregate/')
    print(f'  Figures:   ./figures/<dataset>/')
    print(f'  Report:    ./artifacts/reproduction_report.html')
    print()


if __name__ == '__main__':
    main()
