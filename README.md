# ASIKT Official End-to-End Reproduction (Unified Protocol)

Reproduction of the released [ASIKT](https://github.com/mrsser/ASIKT)
implementation (commit `db8eb4cac6bcf9138b2179e8f279b93a49de2832`) under the
project's **unified question-window evaluation protocol**, so the result is
directly comparable with the project's own models on the same data split,
checkpoint rule, and metric.

This is **not** the score on the author's embedded ASSIST2009 split. The
author split and the project split use different question/concept
dictionaries and different train/validation/test handling (details in
[results/REPRODUCTION.md](results/REPRODUCTION.md)).

## Result (ASSIST2009, fold 0, seed 42)

| Metric | Value |
| --- | ---: |
| `windowauclate_mean` (reported) | 0.7815264028 |
| `windowacclate_mean` | 0.7362824086 |
| `windowaucconcepts` | 0.7697830757 |
| `windowaccconcepts` | 0.7304870560 |
| `testauc` | 0.8434635667 |
| `testacc` | 0.7766218661 |
| `window_testauc` | 0.8453700188 |
| `window_testacc` | 0.7779980658 |
| Best validation AUC | 0.8507974632 (epoch 6) |

Same-protocol comparison with the project model `r316_transfer_ledger`
(same fold 0 / seed 42 / unified AS09 question-window files):

| Model | `windowauclate_mean` | `windowacclate_mean` |
| --- | ---: | ---: |
| Official ASIKT | 0.7815264028 | 0.7362824086 |
| Project `r316_transfer_ledger` | 0.7881642411 | 0.7310130207 |

Each model keeps its own training procedure and hyperparameters; only the
data split, checkpoint selection rule, and question-window evaluation are
shared.

## Protocol

- Dataset: `assist2009_corrected_collapsed` (17,737 questions, 123 concepts)
- Fold 0: train folds 1-4 (3,016 sequences), validation fold 0 (741 sequences),
  test (904 sequences); max sequence length 200
- ASIKT priors estimated **only from train folds** (fold-0 responses excluded)
- Evaluation chain: `evaluate_question -> group_fusion -> effective_fusion
  -> late_fusion` (native pyKT question-window metric)
- Reported metric: `windowauclate_mean`
- Question-window test rows: 61,882; unique test questions: 52,378

## Run configuration

- Seed 42, batch size 32, epochs allowed 300, early-stop patience 7
  (best epoch 6, 14 epochs completed)
- Adam lr 5e-4, betas (0.9, 0.999), eps 1e-8
- Model: `d_model=128`, `dropout=0.2`, `n_blocks=2`, `rasch=1`,
  `final_fc_dim=512`, `n_heads=8`, `d_ff=1024`, `theta=0.5`
- Environment: Python 3.10.18, PyTorch 2.2.0, CUDA 11.8,
  NVIDIA RTX 3060 Laptop (6 GB)

## Reproduce

The integration files keep their original pyKT-toolkit-relative paths under
[integration/](integration/). Place them into your pyKT-toolkit fork and run:

```powershell
python examples\train_asikt_official.py --dataset assist2009 --fold 0 --seed 42 `
  --batch-size 32 --num-epochs 300 --patience 7 --learning-rate 0.0005 `
  --d-model 128 --dropout 0.2 --n-blocks 2 --rasch 1 --final-fc-dim 512 `
  --n-heads 8 --d-ff 1024 --theta 0.5 `
  --output-dir <your-output-dir>
```

`train_asikt_official_loop.py` runs the released `main/run` training and
test functions end to end.

### Deviation from official defaults

The official default batch size is 48. The first 48-batch attempt exhausted
memory on the 6 GB GPU and was not persisted; the reported run uses
batch size 32 (recorded in `results/run_manifest.json`).

## Repository layout

```
integration/
  examples/            train_asikt_official.py, train_asikt_official_loop.py
  pykt/models/         asikt_official.py (bridge), asikt.py, asikt_mamba2.py,
                       init_model.py, evaluate_model.py,
                       asikt_official_source/ (byte-for-byte official release)
  pykt/datasets/       asikt_utils.py, init_dataset.py
  tests/               test_asikt_official.py
results/
  REPRODUCTION.md      full reproduction write-up
  result.json          metrics
  run_manifest.json    full run record (source SHA256, hyperparameters,
                       data files, environment, deviations)
  SHA256SUMS.json      checksums of every artifact
  training_history.json, asikt_priors.pt, test_question_window_arrays.npz,
  qid_test_predictions.txt, qid_test_question_window_predictions.txt,
  train_console.log
```

## Verification

- Official release files are retained byte-for-byte and cross-checked by
  SHA256 (see `results/SHA256SUMS.json` and `results/run_manifest.json`)
- `pytest tests/test_asikt_official.py -q`: 4 passed
- Official difficulty-mask construction matches the bridge to a maximum
  absolute error of `1.2847428787488013e-7`
- Native pyKT question-window metrics and an independent recomputation from
  `test_question_window_arrays.npz` match exactly
- The best checkpoint (69 MB) is not included in this repository to keep it
  light; it is reproducible from the command above.

## License

The official release files under
`integration/pykt/models/asikt_official_source/` carry their own license
(see `LICENSE` inside that directory).
