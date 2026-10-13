from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .constants import COMPETITION_SLUG, DEFAULT_RAW_DIR, DEFAULT_REPORT_PATH
from .data_contract import inspect_data_contract, write_report
from .four_branch_benchmark import run_four_branch_benchmark
from .global_gnn import run_global_gnn
from .hybrid_candidate import build_constant_prr_hybrid
from .lockbox_candidate import run_lockbox_candidate
from .model_audit import build_model_audit, write_model_audit
from .modeling import train_candidate
from .oof import run_smoke_oof
from .prr_oof import run_prr_oof
from .prr_pytorch import compare_prr_methods, finalize_existing_prr_comparison
from .prr_rupture_v2 import run_prr_rupture_v2
from .prr_tail_dependency import finalize_prr_tail_dependency, run_prr_tail_dependency
from .row_activity_benchmark import (
    finalize_row_activity_candidate,
    run_row_activity_benchmark,
)
from .rupture_benchmark import run_rupture_benchmark
from .severity_priors_benchmark import (
    finalize_severity_priors_candidate,
    run_severity_priors_benchmark,
)
from .submission import validate_submission
from .submission_gate import (
    create_reference_registry,
    decision_as_dict,
    evaluate_submission_gate,
    load_json,
    promote_candidate,
    write_json,
)
from .validation import build_drug_disjoint_holdout, build_pair_grouped_folds


def _download_data(raw_dir: Path, force: bool) -> int:
    import kagglehub

    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = kagglehub.competition_download(
            COMPETITION_SLUG,
            output_dir=str(raw_dir),
            force_download=force,
        )
    except Exception as error:  # Kaggle emits several auth/consent exception types.
        print(
            "Téléchargement impossible. Vérifiez que vous avez accepté les règles et lancé "
            "`kaggle auth login` dans cet environnement.\n"
            f"Détail Kaggle : {error}",
            file=sys.stderr,
        )
        return 2
    print(f"Données officielles téléchargées dans : {path}")
    report = inspect_data_contract(raw_dir)
    write_report(report, DEFAULT_REPORT_PATH)
    print(report.to_json())
    return 0 if report.ok else 1


def _inspect_data(raw_dir: Path, report_path: Path | None) -> int:
    report = inspect_data_contract(raw_dir)
    if report_path is not None:
        write_report(report, report_path)
        print(f"Rapport écrit dans {report_path}")
    print(report.to_json())
    return 0 if report.ok else 1


def _validate_submission(candidate: Path, sample: Path) -> int:
    errors = validate_submission(candidate, sample)
    if errors:
        for error in errors:
            print(f"ERREUR: {error}", file=sys.stderr)
        return 1
    print("Soumission valide : schéma, identifiants et domaines de valeurs conformes.")
    return 0


def _build_validation(
    raw_dir: Path,
    output_dir: Path,
    n_splits: int,
    seed: int,
    strict_seeds: list[int],
    strict_fraction: float,
) -> int:
    train = pd.read_csv(raw_dir / "train.csv", low_memory=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, audit = build_pair_grouped_folds(train, n_splits=n_splits, seed=seed)
    manifest.to_csv(output_dir / "pair_grouped_folds.csv", index=False)
    write_json(audit, output_dir / "pair_grouped_audit.json")

    strict_audits = []
    for strict_seed in strict_seeds:
        strict_manifest, strict_audit = build_drug_disjoint_holdout(
            train,
            seed=strict_seed,
            holdout_fraction=strict_fraction,
        )
        strict_manifest.to_csv(output_dir / f"drug_disjoint_seed{strict_seed}.csv", index=False)
        write_json(strict_audit, output_dir / f"drug_disjoint_seed{strict_seed}_audit.json")
        strict_audits.append(strict_audit)

    summary = {"primary": audit, "strict_holdouts": strict_audits}
    write_json(summary, output_dir / "validation_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if audit["passed"] and all(item["passed"] for item in strict_audits) else 1


def _run_smoke(
    raw_dir: Path,
    manifest_path: Path,
    audit_path: Path,
    output_dir: Path,
    binary_threshold: float,
) -> int:
    train = pd.read_csv(raw_dir / "train.csv", low_memory=False)
    manifest = pd.read_csv(manifest_path)
    audit = load_json(audit_path)
    metrics = run_smoke_oof(
        train,
        manifest,
        audit,
        output_dir=output_dir,
        binary_prevalence_threshold=binary_threshold,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


def _init_gate_reference(
    metrics_path: Path,
    registry_path: Path,
    min_improvement: float,
    force: bool,
) -> int:
    if registry_path.exists() and not force:
        print(
            f"Registre déjà présent : {registry_path}. Utilisez --force pour le remplacer.",
            file=sys.stderr,
        )
        return 2
    registry = create_reference_registry(load_json(metrics_path), min_improvement=min_improvement)
    write_json(registry, registry_path)
    print(json.dumps(registry, indent=2, ensure_ascii=False))
    return 0


def _gate_submission(candidate_path: Path, registry_path: Path) -> int:
    decision = evaluate_submission_gate(load_json(candidate_path), load_json(registry_path))
    print(json.dumps(decision_as_dict(decision), indent=2, ensure_ascii=False))
    return 0 if decision.eligible else 3


def _promote_candidate(candidate_path: Path, registry_path: Path) -> int:
    decision, registry = promote_candidate(load_json(candidate_path), load_json(registry_path))
    print(json.dumps(decision_as_dict(decision), indent=2, ensure_ascii=False))
    if not decision.eligible:
        return 3
    write_json(registry, registry_path)
    print("Référence locale mise à jour. Aucune soumission Kaggle n'a été effectuée.")
    return 0


def _model_audit(raw_dir: Path, output_dir: Path) -> int:
    train = pd.read_csv(raw_dir / "train.csv", low_memory=False)
    test = pd.read_csv(raw_dir / "test.csv", low_memory=False)
    audit = build_model_audit(train, test)
    write_model_audit(audit, output_dir)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


def _train_candidate(config: Path, output_dir: Path, submission: Path) -> int:
    result = train_candidate(
        train_path=DEFAULT_RAW_DIR / "train.csv",
        test_path=DEFAULT_RAW_DIR / "test.csv",
        sample_path=DEFAULT_RAW_DIR / "sample_submission.csv",
        folds_path=Path("artifacts/validation/pair_grouped_folds.csv"),
        folds_audit_path=Path("artifacts/validation/pair_grouped_audit.json"),
        strict_manifest_path=Path("artifacts/validation/drug_disjoint_seed2026.csv"),
        registry_path=Path("artifacts/submission_registry.json"),
        config_path=config,
        output_dir=output_dir,
        submission_path=submission,
    )
    print(json.dumps(result["gate"], indent=2, ensure_ascii=False))
    return 0 if result["gate"]["eligible"] else 3


def _build_hybrid_candidate(output_dir: Path, submission: Path) -> int:
    result = build_constant_prr_hybrid(
        train_path=DEFAULT_RAW_DIR / "train.csv",
        test_path=DEFAULT_RAW_DIR / "test.csv",
        sample_path=DEFAULT_RAW_DIR / "sample_submission.csv",
        folds_path=Path("artifacts/validation/pair_grouped_folds.csv"),
        parent_run_dir=Path("artifacts/runs/multimodal_lgbm_extratrees_v1"),
        output_dir=output_dir,
        submission_path=submission,
        registry_path=Path("artifacts/submission_registry.json"),
    )
    print(json.dumps(result["gate"], indent=2, ensure_ascii=False))
    return 0 if result["gate"]["eligible"] else 3


def _compare_prr_pytorch(config: Path, finalize_only: bool) -> int:
    result = (
        finalize_existing_prr_comparison(config) if finalize_only else compare_prr_methods(config)
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _run_prr_oof(config: Path) -> int:
    result = run_prr_oof(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _run_four_branch_benchmark(config: Path) -> int:
    result = run_four_branch_benchmark(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _run_lockbox_candidate(config: Path, finalize_if_improved: bool) -> int:
    result = run_lockbox_candidate(config, finalize_if_improved=finalize_if_improved)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _run_rupture_benchmark(config: Path) -> int:
    result = run_rupture_benchmark(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _run_global_gnn(config: Path) -> int:
    result = run_global_gnn(config)
    short_result = {key: value for key, value in result.items() if key != "fold_results"}
    print(json.dumps(short_result, indent=2))
    return 0


def _run_prr_rupture_v2(config: Path) -> int:
    result = run_prr_rupture_v2(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _run_prr_tail_dependency(config: Path) -> int:
    result = run_prr_tail_dependency(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _finalize_prr_tail_dependency(config: Path) -> int:
    result = finalize_prr_tail_dependency(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _run_severity_priors_benchmark(config: Path) -> int:
    result = run_severity_priors_benchmark(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _run_row_activity_benchmark(config: Path) -> int:
    result = run_row_activity_benchmark(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _finalize_row_activity_candidate(config: Path) -> int:
    result = finalize_row_activity_candidate(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _finalize_severity_priors_candidate(config: Path) -> int:
    result = finalize_severity_priors_candidate(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deep-pharma")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-data", help="Télécharger les données officielles.")
    download.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    download.add_argument("--force", action="store_true")

    inspect = subparsers.add_parser("inspect-data", help="Vérifier le contrat des CSV officiels.")
    inspect.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    inspect.add_argument("--write-report", type=Path, default=None)

    validate = subparsers.add_parser("validate-submission", help="Vérifier un CSV avant envoi.")
    validate.add_argument("candidate", type=Path)
    validate.add_argument(
        "--sample",
        type=Path,
        default=DEFAULT_RAW_DIR / "sample_submission.csv",
    )

    build_validation = subparsers.add_parser(
        "build-validation", help="Construire et auditer les splits sans fuite."
    )
    build_validation.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    build_validation.add_argument("--output-dir", type=Path, default=Path("artifacts/validation"))
    build_validation.add_argument("--n-splits", type=int, default=5)
    build_validation.add_argument("--seed", type=int, default=2026)
    build_validation.add_argument("--strict-seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    build_validation.add_argument("--strict-fraction", type=float, default=0.25)

    smoke = subparsers.add_parser(
        "run-smoke-oof", help="Tester le pipeline OOF avec un baseline constant fold-local."
    )
    smoke.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    smoke.add_argument(
        "--manifest", type=Path, default=Path("artifacts/validation/pair_grouped_folds.csv")
    )
    smoke.add_argument(
        "--audit", type=Path, default=Path("artifacts/validation/pair_grouped_audit.json")
    )
    smoke.add_argument("--output-dir", type=Path, default=Path("artifacts/runs/smoke_baseline_v1"))
    smoke.add_argument("--binary-threshold", type=float, default=0.20)

    reference = subparsers.add_parser(
        "init-gate-reference", help="Initialiser le seuil local du verrou de soumission."
    )
    reference.add_argument("metrics", type=Path)
    reference.add_argument(
        "--registry", type=Path, default=Path("artifacts/submission_registry.json")
    )
    reference.add_argument("--min-improvement", type=float, default=0.0001)
    reference.add_argument("--force", action="store_true")

    gate = subparsers.add_parser(
        "gate-submission", help="Refuser une soumission sans amélioration OOF comparable."
    )
    gate.add_argument("candidate_metrics", type=Path)
    gate.add_argument("--registry", type=Path, default=Path("artifacts/submission_registry.json"))

    promote = subparsers.add_parser(
        "promote-candidate",
        help="Promouvoir localement un candidat éligible, sans le soumettre.",
    )
    promote.add_argument("candidate_metrics", type=Path)
    promote.add_argument(
        "--registry", type=Path, default=Path("artifacts/submission_registry.json")
    )

    audit_model = subparsers.add_parser(
        "model-audit", help="Produire l'audit approfondi avant modélisation."
    )
    audit_model.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    audit_model.add_argument("--output-dir", type=Path, default=Path("reports"))

    candidate = subparsers.add_parser(
        "train-candidate", help="Entraîner et évaluer la baseline multimodale sans soumission."
    )
    candidate.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    candidate.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/runs/multimodal_lgbm_extratrees_v1")
    )
    candidate.add_argument("--submission", type=Path, default=Path("outputs/submission.csv"))

    hybrid = subparsers.add_parser(
        "build-hybrid-candidate",
        help="Conserver Severity/Binary et sélectionner le meilleur PRR OOF.",
    )
    hybrid.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/runs/multimodal_lgbm_extratrees_constant_prr_v1"),
    )
    hybrid.add_argument("--submission", type=Path, default=Path("outputs/submission.csv"))

    pytorch_prr = subparsers.add_parser(
        "compare-prr-pytorch",
        help="Comparer trois losses PRR sur un split partagé, sans sélectionner sur le test.",
    )
    pytorch_prr.add_argument("--config", type=Path, default=Path("configs/prr_pytorch.yaml"))
    pytorch_prr.add_argument(
        "--finalize-only",
        action="store_true",
        help="Reconstruire les tableaux depuis les artefacts sans relire train/test.",
    )
    prr_oof = subparsers.add_parser(
        "run-prr-oof",
        help="Évaluer les modèles PRR sur tous les folds sans jamais charger le test.",
    )
    prr_oof.add_argument("--config", type=Path, default=Path("configs/prr_oof.yaml"))
    four_branch = subparsers.add_parser(
        "run-four-branch-benchmark",
        help="Comparer calibration Binary, GBDT par cible, factorisation et blend OOF.",
    )
    four_branch.add_argument(
        "--config",
        type=Path,
        default=Path("configs/four_branch_benchmark.yaml"),
    )
    lockbox = subparsers.add_parser(
        "run-lockbox-candidate",
        help="Valider Binary C + PRR blend sur lockbox, puis finaliser si meilleur.",
    )
    lockbox.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lockbox_candidate.yaml"),
    )
    lockbox.add_argument(
        "--finalize-if-improved",
        action="store_true",
        help="Charger le test et produire le CSV uniquement si le lockbox est meilleur.",
    )
    rupture = subparsers.add_parser(
        "run-rupture-benchmark",
        help="Evaluer GNN, fusion identite/structure, experts PRR et stacking OOF.",
    )
    rupture.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rupture_benchmark.yaml"),
    )
    nested_cpu = subparsers.add_parser(
        "run-nested-cpu-optimization",
        help="Optimisation bornee PRR/Binary/Severity, selection interne et comparaison OOF.",
    )
    nested_cpu.add_argument(
        "--config", type=Path, default=Path("configs/nested_cpu_optimization.yaml")
    )
    nested_cpu.add_argument("--finalize-only", action="store_true")
    cdn = subparsers.add_parser(
        "run-prr-cdn", help="CDN PRR avec conditioning OOF interne et convergence sans labels."
    )
    cdn.add_argument("--config", type=Path, default=Path("configs/prr_cdn.yaml"))
    global_gnn = subparsers.add_parser(
        "run-global-gnn",
        help="GNN moléculaire global et multimodal, sélection interne, OOF sans test.",
    )
    global_gnn.add_argument(
        "--config", type=Path, default=Path("configs/global_gnn.yaml")
    )
    prr_rupture_v2 = subparsers.add_parser(
        "run-prr-rupture-v2",
        help="Evaluer residual priors, GBDT, k-NN, factorisation et MoE PRR en OOF.",
    )
    prr_rupture_v2.add_argument(
        "--config",
        type=Path,
        default=Path("configs/prr_rupture_v2.yaml"),
    )
    tail_dependency = subparsers.add_parser(
        "run-prr-tail-dependency",
        help="Evaluer un expert PRR conditionnel des extremes avec graphe d'effets en OOF.",
    )
    tail_dependency.add_argument(
        "--config", type=Path, default=Path("configs/prr_tail_dependency.yaml")
    )
    finalize_tail = subparsers.add_parser(
        "finalize-prr-tail-dependency",
        help="Refit et cree un CSV PRR seulement si les cinq folds OOF progressent.",
    )
    finalize_tail.add_argument(
        "--config", type=Path, default=Path("configs/prr_tail_dependency.yaml")
    )
    severity_priors = subparsers.add_parser(
        "run-severity-priors-benchmark",
        help="Comparer la severite avec priors moleculaires cross-fittes en OOF.",
    )
    severity_priors.add_argument(
        "--config", type=Path, default=Path("configs/severity_priors_benchmark.yaml")
    )
    activity = subparsers.add_parser(
        "run-row-activity-benchmark",
        help="Evaluer une porte binaire au niveau paire sur les memes folds OOF.",
    )
    activity.add_argument(
        "--config", type=Path, default=Path("configs/row_activity_benchmark.yaml")
    )
    finalize_activity = subparsers.add_parser(
        "finalize-row-activity-candidate",
        help="Refit la porte d'activite OOF et retire les effets binaires peu fiables.",
    )
    finalize_activity.add_argument(
        "--config", type=Path, default=Path("configs/row_activity_benchmark.yaml")
    )
    finalize_severity = subparsers.add_parser(
        "finalize-severity-priors-candidate",
        help="Refit la meilleure Severity OOF et remplace uniquement cette colonne du CSV.",
    )
    finalize_severity.add_argument(
        "--config", type=Path, default=Path("configs/severity_priors_benchmark.yaml")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "download-data":
        code = _download_data(args.raw_dir, args.force)
    elif args.command == "inspect-data":
        code = _inspect_data(args.raw_dir, args.write_report)
    elif args.command == "validate-submission":
        code = _validate_submission(args.candidate, args.sample)
    elif args.command == "build-validation":
        code = _build_validation(
            args.raw_dir,
            args.output_dir,
            args.n_splits,
            args.seed,
            args.strict_seeds,
            args.strict_fraction,
        )
    elif args.command == "run-smoke-oof":
        code = _run_smoke(
            args.raw_dir,
            args.manifest,
            args.audit,
            args.output_dir,
            args.binary_threshold,
        )
    elif args.command == "init-gate-reference":
        code = _init_gate_reference(
            args.metrics,
            args.registry,
            args.min_improvement,
            args.force,
        )
    elif args.command == "gate-submission":
        code = _gate_submission(args.candidate_metrics, args.registry)
    elif args.command == "promote-candidate":
        code = _promote_candidate(args.candidate_metrics, args.registry)
    elif args.command == "model-audit":
        code = _model_audit(args.raw_dir, args.output_dir)
    elif args.command == "train-candidate":
        code = _train_candidate(args.config, args.output_dir, args.submission)
    elif args.command == "build-hybrid-candidate":
        code = _build_hybrid_candidate(args.output_dir, args.submission)
    elif args.command == "compare-prr-pytorch":
        code = _compare_prr_pytorch(args.config, args.finalize_only)
    elif args.command == "run-four-branch-benchmark":
        code = _run_four_branch_benchmark(args.config)
    elif args.command == "run-lockbox-candidate":
        code = _run_lockbox_candidate(args.config, args.finalize_if_improved)
    elif args.command == "run-rupture-benchmark":
        code = _run_rupture_benchmark(args.config)
    elif args.command == "run-nested-cpu-optimization":
        from .nested_cpu_optimization import run_nested_cpu_optimization

        run_nested_cpu_optimization(args.config, finalize_only=args.finalize_only)
        code = 0
    elif args.command == "run-prr-cdn":
        from .prr_cdn import run

        run(args.config)
        code = 0
    elif args.command == "run-global-gnn":
        code = _run_global_gnn(args.config)
    elif args.command == "run-prr-rupture-v2":
        code = _run_prr_rupture_v2(args.config)
    elif args.command == "run-prr-tail-dependency":
        code = _run_prr_tail_dependency(args.config)
    elif args.command == "finalize-prr-tail-dependency":
        code = _finalize_prr_tail_dependency(args.config)
    elif args.command == "run-severity-priors-benchmark":
        code = _run_severity_priors_benchmark(args.config)
    elif args.command == "run-row-activity-benchmark":
        code = _run_row_activity_benchmark(args.config)
    elif args.command == "finalize-row-activity-candidate":
        code = _finalize_row_activity_candidate(args.config)
    elif args.command == "finalize-severity-priors-candidate":
        code = _finalize_severity_priors_candidate(args.config)
    else:  # run-prr-oof
        code = _run_prr_oof(args.config)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
