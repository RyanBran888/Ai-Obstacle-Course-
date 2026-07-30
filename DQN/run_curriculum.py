from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from curriculum import FULL_COURSE_HORIZON, make_runner
from DQN.DQN_model import ACTIONS, CHANNEL_NAMES, GLOBAL_NAMES, OBS_DIM, OBSERVATION_SCHEMA
from DQN.DQN_train import Config, pin_auto_device
from preview_maps import export_manifest_site, load_manifest
from room_manifest import save_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train with procedural seed pools")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=FULL_COURSE_HORIZON,
        help=f"episode horizon (minimum {FULL_COURSE_HORIZON})",
    )
    parser.add_argument("--episodes-per-seed", type=int, default=50)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument(
        "--recovery-rounds",
        type=int,
        default=8,
        help="extra rounds after restoring the best failed round",
    )
    parser.add_argument(
        "--recovery-pool-max",
        type=int,
        default=128,
        help="largest adaptive training pool used for validation failures",
    )
    parser.add_argument(
        "--recovery-expansions",
        type=int,
        default=2,
        help="maximum adaptive pool doublings per curriculum pool",
    )
    parser.add_argument(
        "--promotion-passes",
        type=int,
        default=1,
        help="consecutive passing greedy evaluations needed to advance",
    )
    parser.add_argument("--validation-seeds", type=int, default=64)
    parser.add_argument(
        "--test-seeds",
        type=int,
        default=256,
        help="untouched test rooms per stage",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="network and training randomness",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=0,
        help="deterministic source for every staged room split",
    )
    parser.add_argument("--output", default="curriculum_agent.pt")
    parser.add_argument("--graph-output", default="curriculum_training.png")
    parser.add_argument("--manifest-output", default="curriculum_rooms.json")
    parser.add_argument("--report-output", default="curriculum_report.json")
    parser.add_argument(
        "--progress-output",
        default=None,
        help="atomic recovery state (defaults beside --output)",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="continue a compatible recovery state",
    )
    parser.add_argument(
        "--resume-retention-upgrade",
        action="store_true",
        help=(
            "resume one stopped retention-v1 state under retention-v2 rules "
            "and write recovery to a new file"
        ),
    )
    parser.add_argument(
        "--warm-start-progress",
        default=None,
        help=(
            "initialize weights, optimizer, and exploration from an earlier "
            "untested progress state, then revalidate from stage one"
        ),
    )
    parser.add_argument(
        "--extend-stopped-rounds",
        type=int,
        default=0,
        help="explicit extra tuning rounds for an exhausted stopped state",
    )
    parser.add_argument(
        "--maps-output",
        default=None,
        help="optional designer map folder",
    )
    parser.add_argument("--map-cell", type=int, default=12)
    parser.add_argument(
        "--include-test-maps",
        action="store_true",
        help="render held-out test maps after final testing",
    )
    parser.add_argument(
        "--plot-every",
        type=int,
        default=500,
        help="refresh the live dashboard every N training episodes",
    )
    parser.add_argument(
        "--plot-max-points",
        type=int,
        default=2_000,
        help="maximum training points drawn in the live dashboard",
    )
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="spend the untouched test set after a successful run",
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="save the dashboard without opening a live window",
    )
    return parser.parse_args()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent.parent
    paths = sorted(
        path
        for path in (
            *list((root / "DQN").glob("*.py")),
            *list((root / "Architecture" / "coop_env").rglob("*.py")),
        )
        if not path.name.startswith("._")
    )
    return {
        str(path.relative_to(root)): _file_sha256(path)
        for path in paths
    }


def _write_report(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"{target} already exists; final reports are not overwritten")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _save_agent(agent, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    agent.save(str(temporary))
    temporary.replace(path)


def _load_progress_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"{path} is not a compatible curriculum progress state")
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError(f"{path} has no curriculum recovery contract")
    if payload.get("contract_sha256") != _payload_sha256(contract):
        raise ValueError(f"{path} has a corrupted curriculum recovery contract")
    return payload


def _warm_start_metadata(
    path: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    status = str(payload.get("status"))
    if status not in {"training", "stopped", "completed"}:
        raise ValueError(
            "warm starts require a training state that has not opened final testing"
        )
    if payload.get("test_results") or payload.get("test_model_sha256"):
        raise ValueError("warm-start progress must not contain final-test data")
    trainer = payload.get("trainer")
    if not isinstance(trainer, dict):
        raise ValueError("warm-start progress has no trainer state")
    return {
        "path": str(path.resolve()),
        "file_sha256": _file_sha256(path),
        "source_contract_sha256": payload.get("contract_sha256"),
        "status": status,
        "completed_pool_gates": len(payload.get("results", ())),
        "trainer_episodes": int(trainer.get("episodes", 0)),
        "trainer_env_steps": int(trainer.get("env_steps", 0)),
        "trainer_updates": int(trainer.get("updates", 0)),
    }


def _retention_upgrade_metadata(
    path: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    status = str(payload.get("status"))
    if status != "stopped":
        raise ValueError(
            "retention upgrades require a stopped curriculum state"
        )
    if not isinstance(payload.get("active"), dict):
        raise ValueError(
            "retention-upgrade progress must contain an active pool"
        )
    if payload.get("test_results"):
        raise ValueError(
            "retention-upgrade progress must not contain final-test results"
        )
    if payload.get("test_model_sha256") is not None:
        raise ValueError(
            "retention-upgrade progress must not contain a final-test model"
        )
    contract_sha256 = payload.get("contract_sha256")
    if not isinstance(contract_sha256, str) or not contract_sha256:
        raise ValueError(
            "retention-upgrade progress has no source contract hash"
        )
    results = payload.get("results", ())
    if not isinstance(results, (list, tuple)):
        raise ValueError(
            "retention-upgrade progress has invalid curriculum results"
        )
    return {
        "kind": "retention_v2",
        "path": str(path.resolve()),
        "file_sha256": _file_sha256(path),
        "source_contract_sha256": contract_sha256,
        "completed_pool_gates": len(results),
        "original_status": status,
    }


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.output).expanduser()
    graph_path = Path(args.graph_output).expanduser()
    manifest_path = Path(args.manifest_output).expanduser()
    report_path = Path(args.report_output).expanduser()
    maps_path = Path(args.maps_output).expanduser() if args.maps_output else None
    resume_path = (
        Path(args.resume_from).expanduser() if args.resume_from else None
    )
    warm_start_path = (
        Path(args.warm_start_progress).expanduser()
        if args.warm_start_progress
        else None
    )
    if args.resume_retention_upgrade and warm_start_path is not None:
        raise ValueError(
            "--resume-retention-upgrade cannot be combined with "
            "--warm-start-progress"
        )
    if args.resume_retention_upgrade and resume_path is None:
        raise ValueError(
            "--resume-retention-upgrade requires --resume-from"
        )
    if (
        args.resume_retention_upgrade
        and args.extend_stopped_rounds <= 0
    ):
        raise ValueError(
            "--resume-retention-upgrade requires "
            "--extend-stopped-rounds greater than zero"
        )
    if resume_path is not None and warm_start_path is not None:
        raise ValueError(
            "--warm-start-progress starts a new run and cannot be combined "
            "with --resume-from"
        )
    if args.extend_stopped_rounds and resume_path is None:
        raise ValueError("--extend-stopped-rounds requires --resume-from")
    progress_path = (
        Path(args.progress_output).expanduser()
        if args.progress_output
        else (
            resume_path
            if resume_path is not None
            else checkpoint.with_name(f"{checkpoint.stem}.progress.pt")
        )
    )
    if (
        args.resume_retention_upgrade
        and resume_path is not None
        and progress_path.resolve() == resume_path.resolve()
    ):
        raise ValueError(
            "--resume-retention-upgrade requires --progress-output to use "
            "a different path from --resume-from"
        )
    if args.map_cell < 4:
        raise ValueError("--map-cell must be at least 4")
    artifacts = {
        "checkpoint": checkpoint,
        "graph": graph_path,
        "manifest": manifest_path,
        "report": report_path,
        "progress": progress_path,
    }
    normalized: dict[Path, str] = {}
    for label, path in artifacts.items():
        resolved = path.resolve()
        if resolved in normalized:
            raise ValueError(
                f"{label} output must differ from {normalized[resolved]} output"
            )
        normalized[resolved] = label
        if path.exists() and not path.is_file():
            raise IsADirectoryError(f"{path} is not a file")
        path.parent.mkdir(parents=True, exist_ok=True)
    if (
        resume_path is not None
        and resume_path.resolve() in normalized
        and normalized[resume_path.resolve()] != "progress"
    ):
        raise ValueError("the resume input cannot also be another output")
    if maps_path is not None:
        if maps_path.resolve() in normalized:
            raise ValueError("maps output must differ from file outputs")
        if maps_path.exists():
            raise FileExistsError(
                f"{maps_path} already exists; choose a new --maps-output"
            )
        maps_path.parent.mkdir(parents=True, exist_ok=True)
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"recovery state does not exist: {resume_path}")
    if warm_start_path is not None and not warm_start_path.is_file():
        raise FileNotFoundError(
            f"warm-start progress does not exist: {warm_start_path}"
        )
    if (
        warm_start_path is not None
        and warm_start_path.resolve() in normalized
    ):
        raise ValueError("warm-start progress cannot also be an output")
    if (
        progress_path.exists()
        and (
            resume_path is None
            or progress_path.resolve() != resume_path.resolve()
        )
    ):
        raise FileExistsError(
            f"{progress_path} already exists; choose a new --progress-output"
        )
    if args.resume_retention_upgrade:
        for label, path in (
            ("checkpoint", checkpoint),
            ("graph", graph_path),
            ("manifest", manifest_path),
            ("report", report_path),
        ):
            if path.exists():
                raise FileExistsError(
                    f"{path} already exists; retention upgrade {label} "
                    "output must use a new path"
                )
    if checkpoint.exists() and resume_path is None:
        raise FileExistsError(f"{checkpoint} already exists; choose a new --output")
    if manifest_path.exists() and resume_path is None:
        raise FileExistsError(
            f"{manifest_path} already exists; choose a new --manifest-output"
        )
    if report_path.exists():
        raise FileExistsError(
            f"{report_path} already exists; choose a new --report-output"
        )
    source_hashes = _source_hashes()
    warm_start_payload = (
        _load_progress_payload(warm_start_path)
        if warm_start_path is not None
        else None
    )
    resume_payload = (
        _load_progress_payload(resume_path)
        if resume_path is not None
        else None
    )
    resume_external: dict[str, Any] = {}
    if resume_payload is not None:
        resume_contract = resume_payload.get("contract")
        if isinstance(resume_contract, dict):
            external = resume_contract.get("external", {})
            if not isinstance(external, dict):
                raise ValueError(
                    "recovery external provenance is invalid"
                )
            resume_external = external
    warm_start_info: dict[str, Any] | None = None
    if warm_start_path is not None and warm_start_payload is not None:
        warm_start_info = _warm_start_metadata(
            warm_start_path,
            warm_start_payload,
        )
    elif resume_payload is not None:
        recovered_warm_start = resume_external.get("warm_start")
        if recovered_warm_start is not None:
            if not isinstance(recovered_warm_start, dict):
                raise ValueError("recovery warm-start provenance is invalid")
            warm_start_info = dict(recovered_warm_start)
    source_upgrade_info: dict[str, Any] | None = None
    if (
        args.resume_retention_upgrade
        and resume_path is not None
        and resume_payload is not None
    ):
        source_upgrade_info = _retention_upgrade_metadata(
            resume_path,
            resume_payload,
        )
    elif resume_payload is not None:
        recovered_source_upgrade = resume_external.get("source_upgrade")
        if recovered_source_upgrade is not None:
            if not isinstance(recovered_source_upgrade, dict):
                raise ValueError(
                    "recovery source-upgrade provenance is invalid"
                )
            source_upgrade_info = dict(recovered_source_upgrade)
    if resume_payload is not None:
        saved_contract = resume_payload.get("contract", {})
        saved_trainer = saved_contract.get("trainer", {})
        saved_request = saved_trainer.get("device")
        if args.device != saved_request:
            raise ValueError(
                f"recovery requires --device {saved_request}; "
                f"received {args.device}"
            )
        saved_device = str(saved_contract.get("device"))
        saved_threads = saved_contract.get("cpu_threads")
        if saved_device != args.device and args.device != "auto":
            raise ValueError("recovery device metadata is inconsistent")
        if args.device == "auto" or saved_device == "cpu":
            pin_auto_device(
                saved_device,
                int(saved_threads) if saved_threads is not None else None,
            )
            print(
                f"Recovery pinned device settings to saved {saved_device}.",
                flush=True,
            )

    cfg = Config(
        max_steps=args.max_steps,
        device=args.device,
        seed=args.seed,
    )
    runner = make_runner(
        cfg,
        validation_size=args.validation_seeds,
        test_size=args.test_seeds,
        episodes_per_seed=args.episodes_per_seed,
        max_rounds=args.max_rounds,
        run_seed=args.seed,
        data_seed=args.data_seed,
        live=not args.no_live,
        plot_every=args.plot_every,
        plot_max_points=args.plot_max_points,
        graph_path=str(graph_path),
        promotion_passes=args.promotion_passes,
        retention_size=args.validation_seeds,
        recovery_rounds=args.recovery_rounds,
        recovery_pool_max=args.recovery_pool_max,
        recovery_expansions=args.recovery_expansions,
        progress_path=str(progress_path),
        resume_from=str(resume_path) if resume_path is not None else None,
        progress_contract={
            "source_sha256": source_hashes,
            "warm_start": warm_start_info,
            "source_upgrade": source_upgrade_info,
        },
        extend_stopped_rounds=args.extend_stopped_rounds,
        retention_upgrade=args.resume_retention_upgrade,
    )
    if warm_start_payload is not None:
        if warm_start_info is None:
            raise RuntimeError("warm-start metadata was not initialized")
        runner.trainer.load_recovery_state(warm_start_payload["trainer"])
        print(
            "Warm-started from "
            f"{warm_start_path} at "
            f"{warm_start_info['trainer_episodes']} prior episodes; "
            "curriculum results and data splits start fresh.",
            flush=True,
        )
    checkpoint_preexisting = checkpoint.is_file()
    finalization_statuses = {"completed", "test_started", "tested"}
    if checkpoint_preexisting:
        if runner.recovery_status not in finalization_statuses:
            raise FileExistsError(
                f"{checkpoint} exists but this recovery state is still training"
            )
        expected_model_hash = runner.model_sha256
        runner.trainer.learners[0].load(str(checkpoint))
        if runner.model_sha256 != expected_model_hash:
            raise ValueError(
                f"{checkpoint} does not match the frozen recovery model"
            )
    if (
        manifest_path.exists()
        and runner.recovery_status not in finalization_statuses
    ):
        raise FileExistsError(
            f"{manifest_path} exists but this recovery state is still training"
        )
    print(f"Training on {runner.trainer.device}")
    if runner.recovery_status == "test_started" and not args.final_test:
        raise ValueError(
            "this recovery state has a sealed partial final test; "
            "resume it with --final-test"
        )
    try:
        results = runner.run()
    except KeyboardInterrupt as error:
        if progress_path.is_file():
            print(
                "No model/report/manifest/test artifacts were finalized. "
                f"Recovery state: {progress_path}"
            )
        else:
            print(
                "Interrupted before the first recovery checkpoint; restart "
                "with the same command."
            )
        raise SystemExit(130) from error
    final = results[-1]
    print(
        f"\nStopped after {final.stage} pool={final.pool_size}; "
        f"promoted={final.promoted}"
    )
    if not runner.completed:
        if final.failure_reasons:
            print("Unmet requirements:")
            for reason in final.failure_reasons:
                print(f"  - {reason}")
        print(
            f"Best evaluated state is recoverable from {progress_path}. "
            "No final checkpoint, report, manifest, maps, or test data were written."
        )
        raise SystemExit(2)
    if not checkpoint_preexisting:
        _save_agent(runner.trainer.learners[0], checkpoint)
    checkpoint_hash = _file_sha256(checkpoint)
    print(f"Frozen checkpoint ready at {checkpoint}")

    if runner.completed and (
        args.final_test
        or runner.recovery_status in {"test_started", "tested"}
    ):
        if _source_hashes() != source_hashes:
            raise RuntimeError("training source files changed during training")
        if len(runner.trainer.learners) != 1:
            raise RuntimeError("final testing requires the shared-network trainer")
        runner.trainer.learners[0].load(str(checkpoint))
        try:
            runner.evaluate_final_test()
        except KeyboardInterrupt as error:
            print(
                "Final test paused at its last completed stage. Resume with "
                f"--resume-from {progress_path} and --final-test."
            )
            raise SystemExit(130) from error
    elif runner.completed:
        print("Final test remains unevaluated; use --final-test only for the chosen run.")
    else:
        print("Final test was not opened because the curriculum did not finish.")

    manifest = runner.prepare_room_manifest()
    manifest_path = save_manifest(manifest, manifest_path)
    manifest_file_hash = _file_sha256(manifest_path)
    print(f"Used room manifest saved to {manifest_path}")

    train_limits = {stage.name: 0 for stage in runner.stages}
    validation_limits = {stage.name: 0 for stage in runner.stages}
    for result in results:
        train_limits[result.stage] = max(
            train_limits[result.stage],
            result.pool_size,
        )
        if result.validation is not None:
            validation_limits[result.stage] = args.validation_seeds
    map_limits = {
        "train": train_limits,
        "validation": validation_limits,
    }
    map_splits = (
        ("train", "validation", "test")
        if runner.test_results and args.include_test_maps
        else ("train", "validation")
    )
    designer_maps = None
    if maps_path is not None:
        manifest_data = load_manifest(manifest_path)
        temporary_maps = Path(
            tempfile.mkdtemp(
                prefix=f".{maps_path.name}.",
                dir=maps_path.parent,
            )
        )
        map_rooms, map_pages = export_manifest_site(
            manifest_path,
            manifest_data,
            temporary_maps,
            splits=map_splits,
            stage_name=None,
            count=None,
            cell=args.map_cell,
            limits=map_limits,
        )
        temporary_maps.replace(maps_path)
        print(
            f"Designer maps saved to {maps_path / 'index.html'} "
            f"({map_rooms} rooms, {map_pages} pages)"
        )
        designer_maps = {
            "path": str(maps_path),
            "index": str(maps_path / "index.html"),
            "splits": list(map_splits),
            "rooms": map_rooms,
            "pages": map_pages,
        }

    if _file_sha256(manifest_path) != manifest_file_hash:
        raise RuntimeError("the saved room manifest changed during the run")
    if _file_sha256(checkpoint) != checkpoint_hash:
        raise RuntimeError("the frozen checkpoint changed during the run")
    if _source_hashes() != source_hashes:
        raise RuntimeError("training source files changed during the run")
    report = {
        "schema_version": 6,
        "curriculum_completed": runner.completed,
        "final_test_requested": (
            args.final_test
            or runner.recovery_status in {"test_started", "tested"}
        ),
        "final_test_evaluated": bool(runner.test_results),
        "run_seed": args.seed,
        "data_seed": args.data_seed,
        "source_sha256": source_hashes,
        "training_coverage": list(runner.training_features),
        "training_config": asdict(cfg),
        "model_contract": {
            "schema": OBSERVATION_SCHEMA,
            "obs_dim": OBS_DIM,
            "actions": list(ACTIONS),
            "channels": list(CHANNEL_NAMES),
            "globals": list(GLOBAL_NAMES),
        },
        "manifest": {
            "path": str(manifest_path),
            "content_sha256": manifest.sha256,
            "file_sha256": manifest_file_hash,
            "schema": manifest.schema_version,
            "selection_algorithm": manifest.selection_algorithm,
            "disjoint_by": [
                "seed",
                "navigation_sha256",
                "task_sha256",
            ],
        },
        "designer_maps": designer_maps,
        "graph": {
            "path": str(graph_path) if graph_path.is_file() else None,
            "scope": (
                "post_resume_segment"
                if resume_path is not None
                else (
                    "warm_started_full_revalidation"
                    if warm_start_info is not None
                    else "full_run"
                )
            ),
        },
        "warm_start": warm_start_info,
        "source_upgrade": source_upgrade_info,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_hash,
        },
        "curriculum": [
            {
                "stage": result.stage,
                "pool_size": result.pool_size,
                "scheduled_pool_size": result.scheduled_pool_size,
                "rounds": result.rounds,
                "recovery_rounds": result.recovery_rounds,
                "best_round": result.best_round,
                "adaptive_expansions": list(result.expansions),
                "promoted": result.promoted,
                "failure_reasons": list(result.failure_reasons),
                "training": result.training.as_dict(),
                "validation": (
                    result.validation.as_dict()
                    if result.validation is not None
                    else None
                ),
                "retention": dict(result.retention),
            }
            for result in results
        ],
        "curriculum_contract": [
            {
                "stage": stage.name,
                "lesson": stage.lesson,
                "objective": stage.objective,
                "pool_sizes": list(stage.pool_sizes or runner.pool_sizes),
                "train_threshold": stage.train_threshold,
                "validation_threshold": stage.validation_threshold,
                "max_wipeout_death_rate": stage.max_wipeout_death_rate,
                "required_features": list(stage.required_features),
            }
            for stage in runner.stages
        ],
        "retention_contract": {
            "validation_rooms_per_prior_stage": runner.retention_size,
            "validation_margin": runner.retention_margin,
            "minimum_success_rate": 0.50,
        },
        "promotion_passes": runner.promotion_passes,
        "recovery": {
            "state": str(progress_path),
            "replay_restored": False,
            "resumed_from": (
                str(resume_path) if resume_path is not None else None
            ),
            "loaded_status": runner.recovery_status,
            "rounds_per_phase": runner.recovery_rounds,
            "maximum_pool_size": runner.recovery_pool_max,
            "maximum_expansions": runner.recovery_expansions,
        },
        "final_test": (
            [
                {
                    "stage": result.stage,
                    "evaluation": result.evaluation.as_dict(),
                    "episodes": [
                        {
                            **episode.as_dict(),
                            "geometry_sha256": room.geometry_sha256,
                            "navigation_sha256": room.navigation_sha256,
                            "task_sha256": room.task_sha256,
                        }
                        for episode, room in zip(
                            result.episodes,
                            manifest.stage(result.stage).test,
                            strict=True,
                        )
                    ],
                }
                for result in runner.test_results
            ]
            if runner.test_results
            else None
        ),
    }
    _write_report(report_path, report)
    print(f"Run report saved to {report_path}")


if __name__ == "__main__":
    main()
