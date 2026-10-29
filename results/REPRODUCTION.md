# ASIKT official end-to-end reproduction

## Scope

This run evaluates the released ASIKT implementation from
<https://github.com/mrsser/ASIKT> on the project's unified ASSIST2009
(`AS09`) fold-0 sequences. The training objective, architecture, optimizer,
and early-stopping rule come from the official implementation. pyKT is used
only for the common question-window evaluation protocol.

This is not a reproduction of the score on the AS09 split embedded in the
author repository. The author split and the project split have different
question/concept dictionaries and different train/validation/test handling.

## Official source

The following files are retained byte-for-byte in
`pykt-toolkit/pykt/models/asikt_official_source`:

Upstream commit:
`db8eb4cac6bcf9138b2179e8f279b93a49de2832`.

| File | SHA256 |
| --- | --- |
| `ASIKT.py` | `9bdbdbb0cbd66576653798f3356bb25ad3976083f3a223981afc3cdd0704bb86` |
| `data_pre.py` | `57e1e44fbc32fb718c5299ad422eb598605a82ce95dcacc109466265baf4f48a` |
| `LICENSE` | `1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6` |
| `main.py` | `84a5162584cee7b4a31791d9f4cb4936dcb3731ce1b2f7124e5334f81489748c` |
| `README.md` | `cb9c030d9ed2e59c9b4ebe5ab942bf4ad5e5a994faef56c5f079e6c3f02641a5` |
| `transformer.py` | `283c69e6dd63b5be466b50967a1f317a1ed0ffa832ada028bbf9725a47d26be5` |
| `mamba2.py` | `ced19be727a107b5e2526ec89f05892f5b73b2f0a63517cfff9c12f18cc59c0d` |
| `run.py` | `45cc7736bcbd884470b59c82a5c96590c4214f0e8fe41bc769c904ef3d88a112` |

`pykt/models/asikt_official.py` is a narrow tensor and prediction alignment
bridge. It instantiates the official `ASIKT` module and backpropagates the
official masked BCE loss. No second BCE loss is added.

## Unified data protocol

- Dataset: project `assist2009_corrected_collapsed`
- Fold: 0
- Train: folds 1-4, 3,016 sequences
- Validation: fold 0, 741 sequences
- Test: 904 sequences
- Dictionary: 17,737 questions and 123 concepts
- Maximum sequence length: 200
- ASIKT priors: estimated only from folds 1-4; fold-0 responses are excluded
- Test question-window rows: 61,882
- Unique test questions: 52,378
- Evaluation: `evaluate_question -> group_fusion -> effective_fusion -> late_fusion`

The author repository's embedded AS09 data contains 4,116 students,
302,498 interactions, 9,798 questions, and 107 concepts. Therefore the score
below must not be presented as a number from the author's own split.

## Run configuration

- Seed: 42
- Batch size: 32
- Epochs allowed: 300
- Early-stopping patience: 7
- Best epoch: 6
- Epochs completed: 14
- Learning rate: 0.0005
- Optimizer: Adam, betas `(0.9, 0.999)`, epsilon `1e-8`
- Model: `d_model=128`, `dropout=0.2`, `n_blocks=2`, `rasch=1`
- Head: `final_fc_dim=512`, `n_heads=8`, `d_ff=1024`, `theta=0.5`
- Environment: Python 3.10.18, PyTorch 2.2.0, CUDA 11.8
- GPU: NVIDIA RTX 3060 Laptop, 6 GB

Command:

```powershell
D:\Anaconda\envs\machine_learning\python.exe examples\train_asikt_official.py --dataset assist2009 --fold 0 --seed 42 --batch-size 32 --num-epochs 300 --patience 7 --learning-rate 0.0005 --d-model 128 --dropout 0.2 --n-blocks 2 --rasch 1 --final-fc-dim 512 --n-heads 8 --d-ff 1024 --theta 0.5 --output-dir E:\StudyWork\sota_kt_lab\work\asikt_official_20261030\assist2009_fold0_seed42
```

## Deviation

The official default batch size is 48. The first attempt at 48 exhausted
memory on the 6 GB GPU. That failed attempt was not retained as a separate
log. The reported run uses batch size 32; the change is recorded in
`run_manifest.json`.

## Results

| Metric | Value |
| --- | ---: |
| `windowauclate_mean` | 0.7815264027514466 |
| `windowacclate_mean` | 0.7362824086448508 |
| `windowaucconcepts` | 0.7697830756768421 |
| `windowaccconcepts` | 0.7304870560098251 |
| `testauc` | 0.8434635667099630 |
| `testacc` | 0.7766218661152753 |
| `window_testauc` | 0.8453700188392371 |
| `window_testacc` | 0.7779980657640232 |
| Best validation AUC | 0.8507974631872273 |

## Verification

- `pytest tests/test_asikt_official.py -q` passes 4 tests.
- The official difficulty-mask construction matches the bridge to a maximum
  absolute error of `1.2847428787488013e-7`.
- Native pyKT question-window metrics and an independent recomputation from
  `test_question_window_arrays.npz` match exactly.
- All predictions are finite, and every question has between 1 and 4 windows.

## Same-protocol comparison

The current project model `r316_transfer_ledger` was run at fold 0, seed 42,
on the same unified AS09 question-window files:

| Model | `windowauclate_mean` | `windowacclate_mean` |
| --- | ---: | ---: |
| Official ASIKT | 0.7815264028 | 0.7362824086 |
| Project `r316_transfer_ledger` | 0.7881642411 | 0.7310130207 |

The ASIKT result is 0.0066378383 lower in window AUC and 0.0052693879 higher
in window ACC than this project run. Each model retains its own training
procedure and hyperparameters; only data split, checkpoint selection rule,
and question-window evaluation are common.
