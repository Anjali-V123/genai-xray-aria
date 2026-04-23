#!/usr/bin/env python3
"""
Dr. ARIA — held-out evaluation on IU X-Ray.

Runs the full analyse pipeline on N real (image, report) pairs and computes
reference-based NLG + clinical metrics against the radiologist-written
reports. Aggregates mean / std / median / 95% bootstrap CI across the set.

Place this file inside backend/  (next to app.py) and run it from there:

    cd backend
    pip install pandas            # not in requirements.txt
    python evaluate.py --kaggle-dir /path/to/iu_xray --num-samples 50

Expected layout for --kaggle-dir
    indiana_reports.csv
    indiana_projections.csv
    images/                       (folder of .png files)

Download: https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university

Why reference-based? The metrics in app.py's compute_all_metrics use a
synthetic reference (findings + RAG text), which measures self-consistency.
For a defensible "model performance" number you need to compare generated
reports against real radiologist reports — that is what this script does.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Import from app.py. This triggers model + RAG load (~30s the first time).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import (  # noqa: E402
    extract_image_features,
    get_findings,
    retrieve_knowledge,
    build_groq_report,
    bleu_score,
    rouge_scores,
    meteor_score,
    cider_score,
    radgraph_detail,
    cosine_similarity_texts,
    ALL_DOCUMENTS,
)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_iu_xray_kaggle(kaggle_dir, n_samples, seed=42):
    """Return a list of (image_path, real_report_text, uid) from the Kaggle
    IU X-Ray layout, frontal projections only, reports non-empty."""
    import pandas as pd

    kaggle_dir = Path(kaggle_dir)
    reports_csv = kaggle_dir / "indiana_reports.csv"
    projs_csv = kaggle_dir / "indiana_projections.csv"

    if not reports_csv.exists() or not projs_csv.exists():
        raise FileNotFoundError(
            f"Expected indiana_reports.csv and indiana_projections.csv "
            f"inside {kaggle_dir}"
        )

    # Auto-detect images folder. Kaggle raddar layout nests them as
    # images/images_normalized/*.png, but some zips unpack to images/*.png.
    candidate_dirs = [
        kaggle_dir / "images" / "images_normalized",
        kaggle_dir / "images",
        kaggle_dir / "images_normalized",
    ]
    images_dir = None
    for d in candidate_dirs:
        if d.exists() and any(d.glob("*.png")):
            images_dir = d
            break
    if images_dir is None:
        raise FileNotFoundError(
            f"Could not locate the images folder under {kaggle_dir}. "
            f"Tried: {[str(c) for c in candidate_dirs]}"
        )
    print(f"Using images from: {images_dir}")

    reports = pd.read_csv(reports_csv)
    projs = pd.read_csv(projs_csv)
    merged = projs.merge(reports, on="uid", how="inner")

    # Keep frontal views only (PA/AP/Frontal labels across CSV versions)
    proj_lower = merged["projection"].astype(str).str.lower()
    is_frontal = (
        proj_lower.str.startswith("frontal")
        | proj_lower.str.contains("pa")
        | proj_lower.str.contains("ap")
    )
    merged = merged[is_frontal]

    # Drop rows missing findings AND impression
    merged = merged.dropna(subset=["findings", "impression"], how="all")

    # Sample deterministically
    merged = merged.sample(n=min(n_samples, len(merged)), random_state=seed)

    samples = []
    for _, row in merged.iterrows():
        img_path = images_dir / str(row["filename"])
        if not img_path.exists():
            continue
        findings = str(row.get("findings") or "").strip()
        impression = str(row.get("impression") or "").strip()
        if not findings and not impression:
            continue
        ref_parts = []
        if findings:
            ref_parts.append(f"FINDINGS: {findings}")
        if impression:
            ref_parts.append(f"IMPRESSION: {impression}")
        samples.append((str(img_path), " ".join(ref_parts), str(row["uid"])))
    return samples


def extract_findings_impression(full_report):
    """Extract the FINDINGS + IMPRESSION sections from a hospital-styled
    report. Used for --short-form evaluation so the hypothesis has a
    comparable format/length to IU X-Ray's 2-sentence references.

    Handles both **FINDINGS** markdown headers and plain FINDINGS: prefixes.
    """
    import re
    text = full_report

    # Match: **FINDINGS** ... up to next **HEADER** or end of string
    def grab(section_name):
        patterns = [
            rf"\*\*{section_name}\*\*(.*?)(?=\*\*[A-Z][A-Z ]+\*\*|\Z)",
            rf"(?:^|\n)\s*{section_name}[:\s]*(.*?)(?=\n\s*[A-Z][A-Z ]{{2,}}[:\s]|\Z)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return ""

    findings = grab("FINDINGS")
    impression = grab("IMPRESSION")
    combined = f"{findings} {impression}".strip()
    # Fall back to the whole report if extraction fails (very short reports)
    return combined if len(combined.split()) >= 5 else full_report


def is_leaked(reference_text, rag_docs, prefix_len=120):
    """Best-effort leakage check: flag if the first `prefix_len` chars of the
    reference appear inside any RAG document. Not perfect, but catches the
    common case where the same IU X-Ray report is already indexed."""
    needle = reference_text[:prefix_len].lower().strip()
    if len(needle) < 40:
        return False
    for doc in rag_docs:
        if needle in doc.lower():
            return True
    return False


# ── Single-sample pipeline + scoring ─────────────────────────────────────────

def run_single(image_path, reference_report, hospital="Apollo", mode="Doctor",
               short_form=False):
    """Run the full Dr. ARIA pipeline on one image, then compute every metric
    against the REAL radiologist report (not the synthetic reference used in
    the UI).

    If short_form=True, scoring uses only the FINDINGS + IMPRESSION sections
    extracted from the generated report, so hypothesis format/length matches
    IU X-Ray's terse 2-sentence references. This is a matched-condition
    evaluation — same model output, scored on the relevant subsections."""
    image = Image.open(image_path).convert("RGB")

    image_features = extract_image_features(image)
    findings_list, top_scores = get_findings(image)
    findings_str = ", ".join([f"{n} ({v:.2f})" for n, v in findings_list]) \
                   if findings_list else "chest x-ray normal study"
    knowledge = retrieve_knowledge(findings_str)

    report = build_groq_report(
        hospital, findings_list, top_scores, image_features, knowledge,
        mode, "Eval Subject", "--", "--", "Evaluation"
    )
    full_generated = report["full_report"]
    generated = extract_findings_impression(full_generated) if short_form \
                else full_generated

    bleu = bleu_score(generated, reference_report)
    rouge = rouge_scores(generated, reference_report)
    meteor = meteor_score(generated, reference_report)
    cider = cider_score(generated, reference_report)
    rg = radgraph_detail(generated, reference_report)
    cos = cosine_similarity_texts(generated, reference_report)

    return {
        "bleu1": bleu["bleu1"],
        "bleu2": bleu["bleu2"],
        "bleu3": bleu["bleu3"],
        "bleu4": bleu["bleu4"],
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "meteor": meteor,
        "cider": cider,
        "radgraph_f1": rg["f1"],
        "radgraph_precision": rg["precision"],
        "radgraph_recall": rg["recall"],
        "cosine_similarity": cos,
        "gen_word_count": len(generated.split()),
        "ref_word_count": len(reference_report.split()),
        "num_findings": len(findings_list),
    }


# ── Aggregation ──────────────────────────────────────────────────────────────

def bootstrap_ci(values, n_boot=1000, ci=95, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return float(values.mean()), float(values.mean())
    boots = np.array([
        rng.choice(values, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boots, (100 - ci) / 2))
    hi = float(np.percentile(boots, 100 - (100 - ci) / 2))
    return lo, hi


def summarize(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_keys = [k for k in results[0].keys() if k != "uid"]

    # Per-sample CSV
    per_path = output_dir / "per_sample_metrics.csv"
    with open(per_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_idx", "uid"] + metric_keys)
        writer.writeheader()
        for i, r in enumerate(results):
            row = {"sample_idx": i, "uid": r.get("uid", "")}
            row.update({k: r[k] for k in metric_keys if k in r})
            writer.writerow(row)

    # Summary CSV
    summary_path = output_dir / "summary.csv"
    rows = []
    for k in metric_keys:
        vals = [r[k] for r in results if isinstance(r.get(k), (int, float))]
        if not vals:
            continue
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        md = float(np.median(vals))
        lo, hi = bootstrap_ci(vals)
        rows.append({
            "metric": k, "mean": m, "std": s, "median": md,
            "ci95_lo": lo, "ci95_hi": hi, "n": len(vals),
        })
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["metric", "mean", "std", "median",
                           "ci95_lo", "ci95_hi", "n"])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "metric": r["metric"],
                "mean": f"{r['mean']:.4f}",
                "std": f"{r['std']:.4f}",
                "median": f"{r['median']:.4f}",
                "ci95_lo": f"{r['ci95_lo']:.4f}",
                "ci95_hi": f"{r['ci95_hi']:.4f}",
                "n": r["n"],
            })

    # Console
    print("\n" + "=" * 78)
    print(f"Dr. ARIA evaluation — N = {len(results)} held-out IU X-Ray samples")
    print("=" * 78)
    print(f"{'Metric':<22} {'Mean':>8} {'Std':>8} {'Median':>8} {'95% CI':>20}")
    print("-" * 78)
    for r in rows:
        ci_str = f"[{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}]"
        print(f"{r['metric']:<22} {r['mean']:>8.4f} {r['std']:>8.4f} "
              f"{r['median']:>8.4f} {ci_str:>20}")
    print("=" * 78)
    print(f"Per-sample CSV: {per_path}")
    print(f"Summary CSV:    {summary_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-dir", required=True,
                        help="Folder containing indiana_reports.csv, "
                             "indiana_projections.csv, and images/")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the sample draw (for reproducibility)")
    parser.add_argument("--output-dir", default="./eval_results")
    parser.add_argument("--hospital", default="Apollo",
                        help="Which hospital style to generate with "
                             "(Apollo/PES Hospital/Manipal/Fortis/AIIMS)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between samples (to respect Groq rate limits)")
    parser.add_argument("--skip-leaked", action="store_true",
                        help="Drop samples whose report is already in the RAG corpus")
    parser.add_argument("--short-form", action="store_true",
                        help="Score only the FINDINGS + IMPRESSION sections "
                             "extracted from the generated hospital report, "
                             "so hypothesis length matches IU X-Ray's terse "
                             "reference format. Output files get a '_short' "
                             "suffix so they don't overwrite full-form results.")
    args = parser.parse_args()
    if args.short_form:
        args.output_dir = args.output_dir.rstrip("/\\") + "_short"

    print(f"Loading {args.num_samples} samples from {args.kaggle_dir}...")
    samples = load_iu_xray_kaggle(args.kaggle_dir, args.num_samples, args.seed)
    print(f"Loaded {len(samples)} image-report pairs.\n")

    if args.skip_leaked:
        before = len(samples)
        samples = [(p, r, u) for p, r, u in samples
                   if not is_leaked(r, ALL_DOCUMENTS)]
        dropped = before - len(samples)
        print(f"Leakage filter: {before} -> {len(samples)} "
              f"(dropped {dropped} whose report is already in the RAG corpus)\n")

    if not samples:
        print("No samples to evaluate after filtering. Aborting.")
        return

    results = []
    t0 = time.time()
    for i, (img_path, ref, uid) in enumerate(samples):
        print(f"[{i + 1}/{len(samples)}] {Path(img_path).name} "
              f"(uid={uid})", end=" ... ", flush=True)
        try:
            metrics = run_single(img_path, ref, hospital=args.hospital,
                                 short_form=args.short_form)
            metrics["uid"] = uid
            results.append(metrics)
            print(f"BLEU-4={metrics['bleu4']:.3f}  "
                  f"ROUGE-L={metrics['rougeL']:.3f}  "
                  f"RadGraph-F1={metrics['radgraph_f1']:.3f}")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(args.delay)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed / max(len(results), 1):.1f}s/sample).")

    if not results:
        print("No successful evaluations.")
        return

    summarize(results, args.output_dir)


if __name__ == "__main__":
    main()