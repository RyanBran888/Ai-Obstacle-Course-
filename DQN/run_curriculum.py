from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from curriculum import make_runner
from DQN.DQN_train import Config
from room_manifest import save_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train with procedural seed pools")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--episodes-per-seed", type=int, default=50)
    parser.add_argument("--max-rounds", type=int, default=8)
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
        help="fixed Architecture room benchmark",
    )
    parser.add_argument("--output", default="curriculum_agent.pt")
    parser.add_argument("--graph-output", default="curriculum_training.png")
    parser.add_argument("--manifest-output", default="curriculum_rooms.json")
    parser.add_argument("--report-output", default="curriculum_report.json")
    parser.add_argument("--plot-every", type=int, default=10)
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="spend the untouched test set after a successful run",
    )
    parser.add_argument("--no-live", action="store_true")
    return parser.parse_args()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent.parent
    files = (
        "QN/QN_model.py",
        "QN/QN_train.py",
        "QN/env_bridge.py",
        "QN/curriculum.py",
        "QN/room_manifest.py",
        "Architecture/coop_env/generation/generator.py",
    )
    return {name: _file_sha256(root / name) for name in files}


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


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.output).expanduser()
    graph_path = Path(args.graph_output).expanduser()
    manifest_path = Path(args.manifest_output).expanduser()
    report_path = Path(args.report_output).expanduser()
    artifacts = {
        "checkpoint": checkpoint,
        "graph": graph_path,
        "manifest": manifest_path,
        "report": report_path,
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
    if checkpoint.exists():
        raise FileExistsError(f"{checkpoint} already exists; choose a new --output")
    if report_path.exists():
        raise FileExistsError(
            f"{report_path} already exists; choose a new --report-output"
        )
    source_hashes = _source_hashes()

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
        graph_path=str(graph_path),
    )
    print(f"Training on {runner.trainer.device}")
    manifest = runner.prepare_room_manifest()
    manifest_path = save_manifest(manifest, manifest_path)
    manifest_file_hash = _file_sha256(manifest_path)
    print(f"Staged room manifest saved to {manifest_path}")

    results = runner.run()
    final = results[-1]
    print(
        f"\nStopped after {final.stage} pool={final.pool_size}; "
        f"promoted={final.promoted}"
    )
    runner.trainer.learners[0].save(str(checkpoint))
    checkpoint_hash = _file_sha256(checkpoint)
    print(f"Frozen checkpoint saved to {checkpoint}")

    if runner.completed and args.final_test:
        if _file_sha256(manifest_path) != manifest_file_hash:
            raise RuntimeError("the saved room manifest changed during training")
        if len(runner.trainer.learners) != 1:
            raise RuntimeError("final testing requires the shared-network trainer")
        runner.trainer.learners[0].load(str(checkpoint))
        runner.evaluate_final_test()
    elif runner.completed:
        print("Final test remains unevaluated; use --final-test only for the chosen run.")
    else:
        print("Final test was not opened because the curriculum did not finish.")

    if _file_sha256(manifest_path) != manifest_file_hash:
        raise RuntimeError("the saved room manifest changed during the run")
    if _file_sha256(checkpoint) != checkpoint_hash:
        raise RuntimeError("the frozen checkpoint changed during the run")
    if _source_hashes() != source_hashes:
        raise RuntimeError("training source files changed during the run")
    report = {
        "schema_version": 1,
        "curriculum_completed": runner.completed,
        "final_test_requested": args.final_test,
        "final_test_evaluated": bool(runner.test_results),
        "run_seed": args.seed,
        "data_seed": args.data_seed,
        "source_sha256": source_hashes,
        "training_config": asdict(cfg),
        "manifest": {
            "path": str(manifest_path),
            "content_sha256": manifest.sha256,
            "file_sha256": manifest_file_hash,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_hash,
        },
        "curriculum": [
            {
                "stage": result.stage,
                "pool_size": result.pool_size,
                "rounds": result.rounds,
                "promoted": result.promoted,
                "training": result.training.as_dict(),
                "validation": (
                    result.validation.as_dict()
                    if result.validation is not None
                    else None
                ),
            }
            for result in results
        ],
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
