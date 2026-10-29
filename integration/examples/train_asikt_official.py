"""Train the released ASIKT model with the unified AS09 protocol."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics


TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_ROOT))

from pykt.datasets import init_dataset4train, init_test_datasets
from pykt.models import evaluate, evaluate_question, init_model
from pykt.models.asikt_official import official_source_hashes
from pykt.models.evaluate_model import (
    collect_question_window_arrays,
    question_window_metrics,
)
from pykt.utils import set_seed


EXAMPLES_ROOT = TOOLKIT_ROOT / "examples"
WORKSPACE_ROOT = TOOLKIT_ROOT.parent
MODEL_NAME = "asikt_official"


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


def _batch_from_dataset(dataset, indices):
    data = dataset.dori
    q = data["qseqs"][indices]
    c = data["cseqs"][indices]
    r = data["rseqs"][indices]
    valid = data["validseqs"][indices].bool()
    return {
        "qseqs": q[:, :-1],
        "shft_qseqs": q[:, 1:],
        "cseqs": c[:, :-1],
        "shft_cseqs": c[:, 1:],
        "rseqs": r[:, :-1],
        "shft_rseqs": r[:, 1:],
        "validseqs": valid[:, :-1],
        "shft_validseqs": valid[:, 1:],
    }


def _aligned_targets(data):
    device = data["rseqs"].device
    valid = torch.cat(
        [data["validseqs"][:, :1], data["shft_validseqs"]], dim=1
    ).to(device).bool()
    concepts = torch.cat(
        [data["cseqs"][:, :1], data["shft_cseqs"]], dim=1
    ).to(device)
    responses = torch.cat(
        [data["rseqs"][:, :1], data["shft_rseqs"]], dim=1
    ).to(device)
    concepts = torch.where(
        valid, concepts.long() + 1, torch.zeros_like(concepts.long())
    )
    responses = torch.where(
        valid, responses.long(), torch.zeros_like(responses.long())
    )
    current = torch.where(
        concepts > 0, responses, torch.full_like(responses, -1)
    )
    return current[:, 1:]


def _evaluate_release_metric(model, dataset, batch_size):
    model.eval()
    predictions, targets, loss_sum = [], [], 0.0
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            indices = torch.arange(start, min(start + batch_size, len(dataset)))
            data = _batch_from_dataset(dataset, indices)
            prediction, loss = model(data, return_loss=True)
            target = _aligned_targets(data)
            selected = target >= -0.9
            predictions.append(torch.masked_select(prediction, selected).cpu())
            targets.append(torch.masked_select(target, selected).cpu())
            loss_sum += float(loss.detach().cpu())
    prediction = torch.cat(predictions).numpy()
    target = torch.cat(targets).numpy()
    auc = metrics.roc_auc_score(target, prediction)
    accuracy = metrics.accuracy_score(target, prediction > 0.5)
    log_loss = metrics.log_loss(target, prediction, labels=[0, 1])
    return {
        "auc": float(auc),
        "accuracy": float(accuracy),
        "log_loss": float(log_loss),
        "loss_sum": loss_sum,
    }


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


def _build_manifest(args, data_config, priors, train_dataset, valid_dataset):
    data_dir = Path(data_config["dpath"])
    data_files = {}
    for name in (
        data_config["train_valid_file"],
        data_config["test_file"],
        data_config["test_question_window_file"],
    ):
        path = data_dir / name
        data_files[name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }

    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        }
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "model": MODEL_NAME,
        "fold": args.fold,
        "seed": args.seed,
        "official_release": {
            "url": "https://github.com/mrsser/ASIKT",
            "source_dir": str(
                (TOOLKIT_ROOT / "pykt" / "models" / "asikt_official_source").resolve()
            ),
            "sha256": official_source_hashes(),
        },
        "protocol": {
            "training": "official ASIKT loss, optimizer, architecture and early stopping",
            "validation_checkpoint": "validation AUC",
            "question_window": "pykt evaluate_question -> group_fusion -> late_fusion",
            "reported_metric": "windowauclate_mean",
        },
        "hyperparameters": {
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "early_stopping_patience": args.patience,
            **{
                key: value
                for key, value in _model_config(args).items()
            },
        },
        "prior_source": {
            "folds": sorted(set(data_config["folds"]) - {args.fold}),
            "train_sequences": len(train_dataset),
            "validation_sequences_excluded": len(valid_dataset),
            "sha256": _tensor_sha256(priors),
        },
        "data_files": data_files,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "gpu": gpu,
        },
    }


def run(args):
    set_seed(args.seed)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    with (TOOLKIT_ROOT / "configs" / "data_config.json").open() as handle:
        all_data_config = json.load(handle)
    data_config = copy.deepcopy(all_data_config[args.dataset])
    data_path = Path(data_config["dpath"])
    if not data_path.is_absolute():
        data_config["dpath"] = str((EXAMPLES_ROOT / data_path).resolve())
    data_config["dataset_name"] = args.dataset
    all_data_config = {args.dataset: data_config}

    train_loader, valid_loader, *_ = init_dataset4train(
        args.dataset,
        MODEL_NAME,
        all_data_config,
        args.fold,
        args.batch_size,
    )
    train_dataset = train_loader.dataset
    valid_dataset = valid_loader.dataset
    priors = data_config["asikt_priors"]
    torch.save(priors, output / "asikt_priors.pt")

    manifest = _build_manifest(
        args, data_config, priors, train_dataset, valid_dataset
    )
    with (output / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    model = init_model(
        MODEL_NAME,
        _model_config(args),
        data_config,
        args.emb_type,
        args.dataset,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    checkpoint_path = output / f"{args.emb_type}_best.ckpt"
    best_auc, best_epoch, best_valid = 0.0, -1, None
    history = []
    started = time.time()

    for epoch in range(1, args.num_epochs + 1):
        epoch_started = time.time()
        model.train()
        indices = np.arange(len(train_dataset))
        np.random.shuffle(indices)
        train_loss = 0.0
        for start in range(0, len(indices), args.batch_size):
            batch_indices = torch.from_numpy(
                indices[start:start + args.batch_size]
            )
            data = _batch_from_dataset(train_dataset, batch_indices)
            optimizer.zero_grad()
            _, loss = model(data, return_loss=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu())

        valid = _evaluate_release_metric(model, valid_dataset, args.batch_size)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss_sum": train_loss,
            "valid_auc": valid["auc"],
            "valid_accuracy": valid["accuracy"],
            "valid_log_loss": valid["log_loss"],
            "seconds": time.time() - epoch_started,
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch}: valid_auc={valid['auc']:.6f}, "
            f"valid_acc={valid['accuracy']:.6f}, "
            f"train_loss_sum={train_loss:.4f}, "
            f"seconds={epoch_metrics['seconds']:.2f}",
            flush=True,
        )
        if valid["auc"] > best_auc:
            best_auc = valid["auc"]
            best_epoch = epoch
            best_valid = valid
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "valid_auc": best_auc,
                },
                checkpoint_path,
            )
        if epoch - best_epoch > args.patience:
            break

    training_seconds = time.time() - started
    with (output / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
        handle.write("\n")

    reloaded = init_model(
        MODEL_NAME,
        _model_config(args),
        data_config,
        args.emb_type,
        args.dataset,
    )
    checkpoint = torch.load(checkpoint_path, map_location=model.model.q_embed.weight.device)
    reloaded.load_state_dict(checkpoint["model_state_dict"])
    reloaded.eval()

    test_loader, test_window_loader, _, question_window_loader = init_test_datasets(
        data_config,
        MODEL_NAME,
        args.batch_size,
    )
    testauc, testacc = evaluate(
        reloaded,
        test_loader,
        MODEL_NAME,
        save_path=str(output / f"{args.emb_type}_test_predictions.txt"),
    )
    window_testauc, window_testacc = evaluate(
        reloaded,
        test_window_loader,
        MODEL_NAME,
        save_path=str(output / f"{args.emb_type}_test_window_predictions.txt"),
    )
    question_aucs, question_accs = evaluate_question(
        reloaded,
        question_window_loader,
        MODEL_NAME,
        ["late_fusion"],
        str(output / f"{args.emb_type}_test_question_window_predictions.txt"),
    )

    qidxs, targets, predictions = collect_question_window_arrays(
        reloaded, question_window_loader
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

    result = {
        "dataset": args.dataset,
        "model": MODEL_NAME,
        "fold": args.fold,
        "seed": args.seed,
        "protocol": "native_pykt_question_window",
        "evaluation_chain": (
            "evaluate_question -> group_fusion -> effective_fusion -> late_fusion"
        ),
        "windowauclate_mean": float(question_aucs["late_mean"]),
        "windowacclate_mean": float(question_accs["late_mean"]),
        "windowaucconcepts": float(question_aucs["concepts"]),
        "windowaccconcepts": float(question_accs["concepts"]),
        "independent_windowauclate_mean": float(
            independent_aucs["late_mean"]
        ),
        "independent_windowacclate_mean": float(
            independent_accs["late_mean"]
        ),
        "testauc": float(testauc),
        "testacc": float(testacc),
        "window_testauc": float(window_testauc),
        "window_testacc": float(window_testacc),
        "best_epoch": int(best_epoch),
        "best_valid": best_valid,
        "training_seconds": training_seconds,
        "question_window_rows": int(len(qidxs)),
        "checkpoint": str(checkpoint_path),
        "gpu_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else None
        ),
    }
    with (output / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="assist2009")
    parser.add_argument("--fold", type=int, choices=[0, 1, 2, 3, 4], default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=7)
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
    parser.add_argument(
        "--output-dir",
        default=str(
            WORKSPACE_ROOT
            / "work"
            / "asikt_official_20261030"
            / "assist2009_fold0_seed42"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
