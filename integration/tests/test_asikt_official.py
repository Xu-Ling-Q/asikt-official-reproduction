import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pykt.datasets import init_dataset4train
from pykt.datasets.asikt_utils import build_asikt_priors
from pykt.models.asikt import build_difficulty_masks
from pykt.models.asikt_official import (
    ASIKTOfficial,
    SOURCE_DIR,
    official_source_hashes,
)
from pykt.models.init_model import init_model


EXPECTED_HASHES = {
    "ASIKT.py": "9bdbdbb0cbd66576653798f3356bb25ad3976083f3a223981afc3cdd0704bb86",
    "data_pre.py": "57e1e44fbc32fb718c5299ad422eb598605a82ce95dcacc109466265baf4f48a",
    "LICENSE": "1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6",
    "main.py": "84a5162584cee7b4a31791d9f4cb4936dcb3731ce1b2f7124e5334f81489748c",
    "README.md": "cb9c030d9ed2e59c9b4ebe5ab942bf4ad5e5a994faef56c5f079e6c3f02641a5",
    "transformer.py": "283c69e6dd63b5be466b50967a1f317a1ed0ffa832ada028bbf9725a47d26be5",
    "mamba2.py": "ced19be727a107b5e2526ec89f05892f5b73b2f0a63517cfff9c12f18cc59c0d",
    "run.py": "45cc7736bcbd884470b59c82a5c96590c4214f0e8fe41bc769c904ef3d88a112",
}


def _priors(num_q=8, num_c=4):
    result = {
        "question_bacc": torch.full((num_q + 1,), 0.6),
        "question_frequency": torch.full((num_q + 1,), 10.0),
        "concept_bacc": torch.full((num_c + 1,), 0.6),
        "concept_frequency": torch.full((num_c + 1,), 10.0),
    }
    for key in result:
        result[key][0] = 0
    return result


def _official_data_module():
    path = SOURCE_DIR / "data_pre.py"
    spec = importlib.util.spec_from_file_location(
        "asikt_release_data_pre_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sequence_row(fold, uid, responses):
    length = 5
    questions = [0, 2, 3, 1, 0] + [-1] * (200 - length)
    concepts = [0, 1, 2, 1, 0] + [-1] * (200 - length)
    answers = list(responses) + [-1] * (200 - length)
    select = [1] * length + [-1] * (200 - length)
    return {
        "fold": fold,
        "uid": uid,
        "questions": ",".join(map(str, questions)),
        "concepts": ",".join(map(str, concepts)),
        "responses": ",".join(map(str, answers)),
        "selectmasks": ",".join(map(str, select)),
    }


class ASIKTOfficialTests(unittest.TestCase):
    def test_preserved_release_hashes_and_factory(self):
        self.assertEqual(official_source_hashes(), EXPECTED_HASHES)
        model = init_model(
            "asikt_official",
            {
                "d_model": 64,
                "d_ff": 64,
                "n_blocks": 1,
                "n_heads": 4,
                "dropout": 0,
            },
            {"num_q": 8, "num_c": 4, "asikt_priors": _priors()},
            "qid",
        )
        self.assertIsInstance(model, ASIKTOfficial)
        self.assertEqual(model.model_name, "asikt_official")
        self.assertEqual(model.model.__class__.__module__, "ASIKT")

    def test_difficulty_masks_match_released_preprocessor(self):
        data_pre = _official_data_module()
        rng = np.random.default_rng(7)
        batch, length, num_q, num_c = 2, 17, 8, 4
        q = rng.integers(0, num_q, size=(batch, length))
        c = rng.integers(0, num_c, size=(batch, length))
        r = rng.integers(0, 2, size=(batch, length))
        valid = np.zeros((batch, length), dtype=bool)
        valid[:, :11] = True
        raw_q = np.where(valid, q, -1)
        raw_c = np.where(valid, c, -1)
        raw_r = np.where(valid, r, -1)
        priors = build_asikt_priors(
            type(
                "Dataset",
                (),
                {
                    "dori": {
                        "qseqs": torch.from_numpy(raw_q),
                        "cseqs": torch.from_numpy(raw_c),
                        "rseqs": torch.from_numpy(raw_r),
                    }
                },
            )(),
            num_q,
            num_c,
        )

        q_release = np.where(valid, raw_q + 1, 0).astype(np.float64)
        c_release = np.where(valid, raw_c + 1, 0).astype(np.float64)
        r_release = np.where(valid, raw_r, 0).astype(np.float64)
        _, q_bacc, q_freq, _, _ = data_pre.global_feature_q(
            q_release, r_release
        )
        _, c_bacc, c_freq, _, _ = data_pre.global_feature_kc(
            c_release, r_release
        )
        q_accuracy = np.asarray(
            data_pre.Q_accuracy(q_release, r_release, q_freq, q_bacc)
        )
        c_accuracy = np.asarray(
            data_pre.KC_accuracy(c_release, r_release, c_freq, c_bacc)
        )
        expected_q, expected_qa = data_pre.Q_mask(
            q_release,
            c_release,
            r_release,
            q_accuracy,
            c_accuracy,
            0.5,
        )

        actual_q, actual_qa = build_difficulty_masks(
            torch.from_numpy(q_release).long(),
            torch.from_numpy(c_release).long(),
            torch.from_numpy(r_release).long(),
            torch.from_numpy(valid),
            priors["question_bacc"],
            priors["question_frequency"],
            priors["concept_bacc"],
            priors["concept_frequency"],
            0.5,
        )
        np.testing.assert_allclose(actual_q.numpy(), expected_q, atol=1e-6)
        np.testing.assert_allclose(actual_qa.numpy(), expected_qa, atol=1e-6)

    def test_priors_ignore_validation_fold_responses(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [
                _sequence_row(i, 100 + i, [1, 0, 1, 0, 1])
                for i in range(5)
            ]
            pd.DataFrame(rows).to_csv(root / "base.csv", index=False)
            changed = copy.deepcopy(rows)
            changed[0]["responses"] = ",".join(
                ["0", "1", "0", "1", "0"] + ["-1"] * 195
            )
            pd.DataFrame(changed).to_csv(
                root / "changed.csv", index=False
            )

            def fit(filename):
                config = {
                    "dpath": str(root),
                    "num_q": 4,
                    "num_c": 3,
                    "input_type": ["questions", "concepts"],
                    "train_valid_file": filename,
                    "folds": [0, 1, 2, 3, 4],
                }
                init_dataset4train(
                    "synthetic",
                    "asikt_official",
                    {"synthetic": config},
                    0,
                    2,
                )
                return config["asikt_priors"]

            base = fit("base.csv")
            intervention = fit("changed.csv")
            for key in base:
                torch.testing.assert_close(base[key], intervention[key])

    def test_release_forward_aligns_to_observed_targets(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        num_q, num_c, length = 8, 4, 200
        valid = torch.zeros(1, length, dtype=torch.bool)
        valid[:, :12] = True
        q = torch.randint(0, num_q, (1, length))
        c = torch.randint(0, num_c, (1, length))
        r = torch.randint(0, 2, (1, length)).float()
        data = {
            "qseqs": q[:, :-1],
            "shft_qseqs": q[:, 1:],
            "cseqs": c[:, :-1],
            "shft_cseqs": c[:, 1:],
            "rseqs": r[:, :-1],
            "shft_rseqs": r[:, 1:],
            "validseqs": valid[:, :-1],
            "shft_validseqs": valid[:, 1:],
        }
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        model = ASIKTOfficial(
            num_c,
            num_q,
            _priors(num_q, num_c),
            d_model=64,
            d_ff=64,
            n_blocks=1,
            n_heads=4,
            dropout=0,
        ).to(device)
        prediction, loss = model(data, return_loss=True)
        loss.backward()
        self.assertEqual(tuple(prediction.shape), (1, length - 1))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                parameter.grad is None
                or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
