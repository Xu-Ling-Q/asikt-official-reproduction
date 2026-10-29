"""Run ASIKT through its released main/run training and test functions."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_ROOT))

from pykt.datasets import init_dataset4train, init_test_datasets
from pykt.models import evaluate, evaluate_question, init_model
from pykt.models.asikt import build_difficulty_masks
from pykt.models.asikt_official import official_source_hashes
from pykt.models.evaluate_model import (
    collect_question_window_arrays,
    question_window_metrics,
)
from pykt.utils import set_seed


EXAMPLES_ROOT = TOOLKIT_ROOT / "examples"
WORKSPACE_ROOT = TOOLKIT_ROOT.parent
MODEL_NAME = "asikt_official"
OFFICIAL_COMMIT = "db8eb4cac6bcf9138b2179e8f279b93a49de2832"
SOURCE_DIR = TOOLKIT_ROOT / "pykt" / "models" / "asikt_official_source"


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(values):
    digest = hashlib.sha256()
    for name in sorted(values):
        tensor = values[name].detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _load_official_entrypoints():
    source = str(SOURCE_DIR.resolve())
    sys.path.insert(0, source)
    importlib.invalidate_caches()
    official_main = importlib.import_module("main")
    official_data_pre = importlib.import_module("data_pre")
    official_run = importlib.import_module("run")
    return official_main, official_data_pre, official_run


def _model_config(args):
    return {
        "d_model": args.d_model,
        "dropout": args.dropout,
        "n_blocks": args.n_blocks,
        "rasch": args.rasch,
        "final_fc_dim": args.final_fc_dim,
        "n_heads": args.n_heads,
        "d_ff": args.d_ff,
        "theta": args.theta,
    }


def _official_params(args, data_config):
    return SimpleNamespace(
        max_iter=args.num_epochs,
        train_set=1,
        seed=args.seed,
        optim="adam",
        batch_size=args.batch_size,
        lr=args.learning_rate,
        final_fc_dim=args.final_fc_dim,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
        n_block=args.n_blocks,
        n_head=args.n_heads,
        rasch=args.rasch,
        model="ASIKT",
        dataset=args.dataset,
        seqlen=data_config["maxlen"],
        n_kc=data_config["num_c"],
        n_q=data_config["num_q"],
        save=args.official_save,
        load=args.official_save,
    )


def _dataset_arrays(dataset, priors, chunk_size=256):
    data = dataset.dori
    q_raw = data["qseqs"].long()
    c_raw = data["cseqs"].long()
    r_raw = data["rseqs"].float()
    valid = data["validseqs"].bool()
    zeros = torch.zeros_like(q_raw)
    q = torch.where(valid, q_raw + 1, zeros)
    c = torch.where(valid, c_raw + 1, zeros)
    r = torch.where(valid, r_raw, torch.zeros_like(r_raw))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    question_bacc = priors["question_bacc"].to(device)
    question_frequency = priors["question_frequency"].to(device)
    concept_bacc = priors["concept_bacc"].to(device)
    concept_frequency = priors["concept_frequency"].to(device)

    question_masks, interaction_masks = [], []
    with torch.inference_mode():
        for start in range(0, len(dataset), chunk_size):
            stop = min(start + chunk_size, len(dataset))
            q_chunk = q[start:stop].to(device)
            c_chunk = c[start:stop].to(device)
            r_chunk = r[start:stop].to(device)
            valid_chunk = valid[start:stop].to(device)
            question_mask, interaction_mask = build_difficulty_masks(
                q_chunk,
                c_chunk,
                r_chunk,
                valid_chunk,
                question_bacc,
                question_frequency,
                concept_bacc,
                concept_frequency,
                theta=0.5,
            )
            question_masks.append(question_mask.cpu())
            interaction_masks.append(interaction_mask.cpu())

    masks = (
        torch.cat(question_masks, dim=0).numpy(),
        torch.cat(interaction_masks, dim=0).numpy(),
    )
    arrays = (
        q.cpu().numpy().astype(np.float64, copy=False),
        c.cpu().numpy().astype(np.float64, copy=False),
        r.cpu().numpy().astype(np.float64, copy=False),
    )
    return (*arrays, *masks)


def _parse_official_result(path):
    keys = (
        "valid_auc",
        "train_auc",
        "valid_loss",
        "train_loss",
        "valid_accuracy",
        "train_accuracy",
    )
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    result = {}
    for key in keys:
        marker = f"{key}:"
        index = lines.index(marker)
        result[key] = ast.literal_eval(lines[index + 1])
    return result


def _data_manifest(data_config):
    data_dir = Path(data_config["dpath"])
    result = {}
    for name in (
        data_config["train_valid_file"],
        data_config["test_file"],
        data_config["test_question_window_file"],
    ):
        path = data_dir / name
        result[name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    return result


def _manifest(args, data_config, priors, train_dataset, valid_dataset):
    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": properties.total_memory,
        }
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "model": "ASIKT",
        "model_adapter": MODEL_NAME,
        "fold": args.fold,
        "seed": args.seed,
        "execution": {
            "training_entrypoint": "official main.train_one_dataset -> run.train -> ASIKT.forward",
            "test_entrypoint": "official run.test",
            "question_window_evaluation": "pykt evaluate_question -> group_fusion -> effective_fusion -> late_fusion",
            "official_main_invoked": True,
            "official_train_invoked": True,
            "official_test_invoked": True,
        },
        "official_release": {
            "url": "https://github.com/mrsser/ASIKT",
            "commit": OFFICIAL_COMMIT,
            "source_dir": str(SOURCE_DIR.resolve()),
            "sha256": official_source_hashes(),
        },
        "hyperparameters": {
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            **{key: value for key, value in _model_config(args).items()},
        },
        "official_early_stopping": {
            "condition": "idx - best_epoch > 5",
            "best_epoch_zero_based": "saved by official main.py",
        },
        "prior_source": {
            "folds": sorted(set(data_config["folds"]) - {args.fold}),
            "train_sequences": len(train_dataset),
            "validation_sequences_excluded": len(valid_dataset),
            "sha256": _tensor_sha256(priors),
        },
        "protocol_scope": {
            "name": "project unified AS09 fold 0",
            "sequence_source": "assist2009_corrected_collapsed",
            "fold_0_train_sequences": len(train_dataset),
            "fold_0_validation_sequences": len(valid_dataset),
            "test_sequences": 904,
            "num_questions": data_config["num_q"],
            "num_concepts": data_config["num_c"],
            "max_sequence_length": data_config["maxlen"],
            "uses_author_embedded_as09_split": False,
        },
        "deviations": [
            {
                "parameter": "batch_size",
                "official_default": 48,
                "used": args.batch_size,
                "reason": "Batch 48 exhausted memory on the 6 GB RTX 3060 Laptop GPU.",
            },
            {
                "parameter": "difficulty_priors",
                "official_script": "data_pre.py estimates priors from the full ASSIST2009 file",
                "used": "folds 1-4 only; fold-0 validation responses excluded",
                "reason": "Required for leakage-free comparison under the unified fold-0 protocol.",
            },
        ],
        "data_files": _data_manifest(data_config),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "gpu": gpu,
        },
    }


def run(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)

    set_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with (TOOLKIT_ROOT / "configs" / "data_config.json").open() as handle:
        all_data_config = json.load(handle)
    data_config = copy.deepcopy(all_data_config[args.dataset])
    data_path = Path(data_config["dpath"])
    if not data_path.is_absolute():
        data_config["dpath"] = str((EXAMPLES_ROOT / data_path).resolve())
    data_config["dataset_name"] = args.dataset

    train_loader, valid_loader, *_ = init_dataset4train(
        args.dataset,
        MODEL_NAME,
        {args.dataset: data_config},
        args.fold,
        args.batch_size,
    )
    train_dataset = train_loader.dataset
    valid_dataset = valid_loader.dataset
    priors = data_config["asikt_priors"]
    torch.save(priors, output / "asikt_priors.pt")

    official_main, _, official_run = _load_official_entrypoints()
    params = _official_params(args, data_config)
    print("Building official train/validation difficulty masks", flush=True)
    train_arrays = _dataset_arrays(train_dataset, priors)
    valid_arrays = _dataset_arrays(valid_dataset, priors)
    torch.cuda.empty_cache()

    started = time.time()
    best_epoch = official_main.train_one_dataset(
        params,
        "",
        *train_arrays,
        *valid_arrays,
    )
    training_seconds = time.time() - started

    identifiers = official_main.get_file_name_identifier(params)
    file_name = "".join(str(item[0]) + str(item[1]) for item in identifiers)
    checkpoint_path = (
        output
        / "model"
        / params.model
        / params.save
        / f"{file_name}_{best_epoch}"
    )
    official_result_path = (
        output / "result" / params.model / params.save / file_name
    )
    history = _parse_official_result(official_result_path)
    with (output / "official_training_history.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(history, handle, indent=2)
        handle.write("\n")

    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    official_net = official_main.load_model(params)
    official_net.load_state_dict(checkpoint["model_state_dict"])

    print("Building official test difficulty masks", flush=True)
    test_loader, test_window_loader, _, question_window_loader = init_test_datasets(
        data_config,
        MODEL_NAME,
        args.batch_size,
    )
    test_dataset = test_loader.dataset
    test_arrays = _dataset_arrays(test_dataset, priors)
    torch.cuda.empty_cache()
    test_loss, test_accuracy, test_auc = official_run.test(
        official_net,
        params,
        None,
        *test_arrays,
        process="test",
    )
    del test_arrays
    torch.cuda.empty_cache()

    bridge = init_model(
        MODEL_NAME,
        _model_config(args),
        data_config,
        args.emb_type,
        args.dataset,
    )
    bridge.model.load_state_dict(checkpoint["model_state_dict"])
    bridge.eval()

    pykt_testauc, pykt_testacc = evaluate(
        bridge,
        test_loader,
        MODEL_NAME,
        save_path=str(output / f"{args.emb_type}_test_predictions.txt"),
    )
    window_testauc, window_testacc = evaluate(
        bridge,
        test_window_loader,
        MODEL_NAME,
        save_path=str(output / f"{args.emb_type}_test_window_predictions.txt"),
    )
    question_aucs, question_accs = evaluate_question(
        bridge,
        question_window_loader,
        MODEL_NAME,
        ["late_fusion"],
        str(output / f"{args.emb_type}_test_question_window_predictions.txt"),
    )
    qidxs, targets, predictions = collect_question_window_arrays(
        bridge, question_window_loader
    )
    np.savez_compressed(
        output / "test_question_window_arrays.npz",
        qidxs=qidxs,
        targets=targets,
        predictions=predictions,
    )
    independent_aucs, independent_accs = question_window_metrics(
        qidxs, targets, predictions
    )
    if not np.isclose(
        independent_aucs["late_mean"],
        question_aucs["late_mean"],
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError("Independent question-window AUC did not reproduce pyKT")
    if not np.isclose(
        independent_accs["late_mean"],
        question_accs["late_mean"],
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError("Independent question-window ACC did not reproduce pyKT")

    best_valid = {
        "auc": float(history["valid_auc"][best_epoch]),
        "accuracy": float(history["valid_accuracy"][best_epoch]),
        "log_loss": float(history["valid_loss"][best_epoch]),
    }
    result = {
        "dataset": args.dataset,
        "model": MODEL_NAME,
        "official_model": "ASIKT",
        "official_commit": OFFICIAL_COMMIT,
        "fold": args.fold,
        "seed": args.seed,
        "protocol": "native_pykt_question_window",
        "evaluation_chain": "evaluate_question -> group_fusion -> effective_fusion -> late_fusion",
        "windowauclate_mean": float(question_aucs["late_mean"]),
        "windowacclate_mean": float(question_accs["late_mean"]),
        "windowaucconcepts": float(question_aucs["concepts"]),
        "windowaccconcepts": float(question_accs["concepts"]),
        "independent_windowauclate_mean": float(independent_aucs["late_mean"]),
        "independent_windowacclate_mean": float(independent_accs["late_mean"]),
        "testauc": float(pykt_testauc),
        "testacc": float(pykt_testacc),
        "window_testauc": float(window_testauc),
        "window_testacc": float(window_testacc),
        "official_standard_test": {
            "loss": float(test_loss),
            "accuracy": float(test_accuracy),
            "auc": float(test_auc),
        },
        "best_epoch": int(best_epoch),
        "epochs_completed": len(history["valid_auc"]),
        "best_valid": best_valid,
        "training_seconds": training_seconds,
        "question_window_rows": int(len(qidxs)),
        "unique_questions": int(len(np.unique(qidxs))),
        "checkpoint": str(checkpoint_path),
        "official_result_log": str(official_result_path),
        "gpu_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else None
        ),
    }
    with (output / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    manifest = _manifest(args, data_config, priors, train_dataset, valid_dataset)
    manifest["training_summary"] = {
        "best_epoch": int(best_epoch),
        "epochs_completed": len(history["valid_auc"]),
        "training_seconds": training_seconds,
    }
    with (output / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="assist2009")
    parser.add_argument("--fold", type=int, choices=[0, 1, 2, 3, 4], default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--rasch", type=int, choices=[0, 1], default=1)
    parser.add_argument("--final-fc-dim", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--emb-type", default="qid")
    parser.add_argument("--official-save", default="asikt_unified_as09")
    parser.add_argument(
        "--output-dir",
        default=str(
            WORKSPACE_ROOT
            / "work"
            / "asikt_official_20261030"
            / "assist2009_fold0_seed42_official_loop"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
