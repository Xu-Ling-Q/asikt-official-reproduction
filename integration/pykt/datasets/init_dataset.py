import os, sys
import json

from torch.utils.data import DataLoader
import numpy as np
from .data_loader import KTDataset
from .dkt_forget_dataloader import DktForgetDataset
from .atdkt_dataloader import ATDKTDataset
from .lpkt_dataloader import LPKTDataset
from .lpkt_utils import generate_time2idx
from .que_data_loader import KTQueDataset
from pykt.config import que_type_models
from .dimkt_dataloader import DIMKTDataset
from .que_data_loader_promptkt import KTQueDataset_promptKT
from .mockt_data_loader import MOCKTDataset
from .pretrain_utils import get_pretrain_data
from .r239_data_loader import R239Dataset
from .r291_data_loader import R291Dataset
from .r303_data_loader import R303Dataset
from .asikt_utils import build_asikt_priors

NATIVE_EVIDENCE_MODELS = (
    "r239_evidence_schema",
    "r265_dual_timescale",
    "r290_rank_aligned",
    "r291_conserved_evidence",
    "r292_budgeted_schema",
    "r293_evidence_retrieval",
    "r294_protected_retrieval",
    "r295_source_evidence",
    "r295_temporal_evidence",
    "r295_role_temporal",
    "r296_source_owned",
    "r297_contextual_sources",
    "r298_centered_sources",
    "r299_factored_sources",
    "r300_temporal_sources",
    "r301_aligned_moments",
    "r303_crossfit_reference",
    "r305_path_ablations",
    "r307_grounded_history",
    "r309_write_separation",
    "r310_transfer_supervision",
    "r311_role_ablation",
    "r312_role_separation",
    "r313_orthogonal_residual",
    "r314_selective_residual",
    "r316_transfer_ledger",
)


def init_test_datasets(data_config, model_name, batch_size, diff_level=None, args=None, re_mapping=False, skip_question_window=False):
    dataset_name = data_config["dataset_name"]
    print(f"model_name is {model_name}, dataset_name is {dataset_name}")
    test_question_loader, test_question_window_loader = None, None
    if model_name in ["dkt_forget", "bakt_time", "fa_kt", "mtkt"]:
        test_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
        test_window_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]),
                                        data_config["input_type"], {-1})
        if "test_question_file" in data_config:
            test_question_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_question_file"]), data_config["input_type"], {-1}, True)
            test_question_window_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"], {-1}, True)
    elif model_name in ["lpkt"]:
        print(f"model_name in lpkt")
        at2idx, it2idx = generate_time2idx(data_config)
        test_dataset = LPKTDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]), at2idx, it2idx, data_config["input_type"], {-1})
        test_window_dataset = LPKTDataset(os.path.join(data_config["dpath"], data_config["test_window_file_quelevel"]), at2idx, it2idx, data_config["input_type"], {-1})
        test_question_dataset = None
        test_question_window_dataset= None
    elif model_name in ["rkt"] and dataset_name in ["statics2011", "assist2015", "poj"]:
        test_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
        test_window_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]), data_config["input_type"], {-1})
        if "test_question_file" in data_config:
            test_question_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_question_file"]), data_config["input_type"], {-1}, True)
            test_question_window_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"], {-1}, True)
    elif model_name in que_type_models:
        if model_name not in ["promptkt", "unikt"]:
            test_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
                            input_type=data_config["input_type"], folds=[-1], 
                            concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
            test_window_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_window_file_quelevel"]),
                            input_type=data_config["input_type"], folds=[-1], 
                            concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
        else:
            dataset = data_config["dpath"].split("/")[-1]
            if dataset == "":
                dataset = data_config["dpath"].split("/")[-2]
            if dataset in [
                "assist2009",
                "algebra2005",
                "bridge2algebra2006",
                "nips_task34",
                "ednet",
                "peiyou",
                "ednet5w",
                "ednet_all"
            ]:
                test_dataset = KTQueDataset_promptKT(
                    os.path.join(
                        data_config["dpath"],
                        data_config["test_file_quelevel"],
                    ),
                    input_type=data_config["input_type"],
                    folds=[-1],
                    concept_num=data_config["num_c"],
                    max_concepts=data_config["max_concepts"],
                    dataset_name=args.dataset_name,
                )
                test_path = os.path.join(
                    data_config["dpath"],
                    data_config["test_window_file_quelevel"],
                )
                if not os.path.exists(test_path):
                    print("not exist")
                    sys.exit(1)
                test_window_dataset = KTQueDataset_promptKT(
                    test_path,
                    input_type=data_config["input_type"],
                    folds=[-1],
                    concept_num=data_config["num_c"],
                    max_concepts=data_config["max_concepts"],
                    dataset_name=args.dataset_name,
                )        
        test_question_dataset = None
        test_question_window_dataset= None
    elif model_name in ["atdkt"]:
        test_dataset = ATDKTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
        test_window_dataset = ATDKTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]), data_config["input_type"], {-1})
        if "test_question_file" in data_config:
            test_question_dataset = ATDKTDataset(os.path.join(data_config["dpath"], data_config["test_question_file"]), data_config["input_type"], {-1}, True)
            test_question_window_dataset = ATDKTDataset(os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"], {-1}, True)
    elif model_name in ["dimkt"]:
        test_dataset = DIMKTDataset(data_config["dpath"],os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1}, diff_level=diff_level)
        test_window_dataset = DIMKTDataset(data_config["dpath"],os.path.join(data_config["dpath"], data_config["test_window_file"]), data_config["input_type"], {-1}, diff_level=diff_level)
        if "test_question_file" in data_config:
            test_question_dataset = DIMKTDataset(data_config["dpath"],os.path.join(data_config["dpath"], data_config["test_question_file"]), data_config["input_type"], {-1}, True, diff_level=diff_level)
            test_question_window_dataset = DIMKTDataset(data_config["dpath"],os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"], {-1}, True, diff_level=diff_level)
    elif model_name in ["mockt"]:
        test_dataset = MOCKTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
        test_window_dataset = MOCKTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]), data_config["input_type"], {-1})
        if "test_question_file" in data_config:
            test_question_dataset = MOCKTDataset(os.path.join(data_config["dpath"], data_config["test_question_file"]), data_config["input_type"], {-1}, True)
            test_question_window_dataset = MOCKTDataset(os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"], {-1}, True)
    elif model_name in NATIVE_EVIDENCE_MODELS:
        num_items = max(1, int(data_config["num_q"]) + 1)
        num_concepts = int(data_config["num_c"]) + 1
        dataset_cls = R291Dataset if model_name in ["r291_conserved_evidence", "r292_budgeted_schema", "r293_evidence_retrieval", "r294_protected_retrieval", "r295_source_evidence", "r295_temporal_evidence", "r295_role_temporal", "r296_source_owned", "r297_contextual_sources", "r298_centered_sources", "r299_factored_sources", "r300_temporal_sources", "r301_aligned_moments"] else R239Dataset
        if model_name in ("r303_crossfit_reference", "r305_path_ablations", "r307_grounded_history", "r309_write_separation", "r310_transfer_supervision", "r311_role_ablation", "r312_role_separation"):
            dataset_cls = R303Dataset
        if "r239_priors" not in data_config:
            raise RuntimeError(
                "R239 test evaluation needs the training-fold priors. "
                "Use the same in-memory data_config after init_dataset4train."
            )
        test_dataset = dataset_cls(
            os.path.join(data_config["dpath"], data_config["test_file"]),
            data_config["input_type"],
            {-1},
            num_items,
            num_concepts,
            priors=data_config["r239_priors"],
        )
        test_window_dataset = dataset_cls(
            os.path.join(data_config["dpath"], data_config["test_window_file"]),
            data_config["input_type"],
            {-1},
            num_items,
            num_concepts,
            priors=data_config["r239_priors"],
        )
        question_path = os.path.join(
            data_config["dpath"], data_config.get("test_question_file", "")
        )
        question_window_path = os.path.join(
            data_config["dpath"],
            data_config.get("test_question_window_file", ""),
        )
        test_question_dataset = (
            dataset_cls(
                question_path,
                data_config["input_type"],
                {-1},
                num_items,
                num_concepts,
                priors=data_config["r239_priors"],
                qtest=True,
            )
            if data_config.get("test_question_file")
            and os.path.exists(question_path)
            else None
        )
        test_question_window_dataset = (
            None
            if skip_question_window
            else dataset_cls(
                question_window_path,
                data_config["input_type"],
                {-1},
                num_items,
                num_concepts,
                priors=data_config["r239_priors"],
                qtest=True,
            )
            if data_config.get("test_question_window_file")
            and os.path.exists(question_window_path)
            else None
        )
    else:
        test_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
        test_window_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]), data_config["input_type"], {-1})
        if "test_question_file" in data_config:
            test_question_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_question_file"]), data_config["input_type"], {-1}, True)
            test_question_window_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"], {-1}, True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_window_loader = DataLoader(test_window_dataset, batch_size=batch_size, shuffle=False)
    if "test_question_file" in data_config:
        print(f"has test_question_file!")
        test_question_loader,test_question_window_loader = None,None
        if not test_question_dataset is None:
            test_question_loader = DataLoader(test_question_dataset, batch_size=batch_size, shuffle=False)
        if not test_question_window_dataset is None:
            test_question_window_loader = DataLoader(test_question_window_dataset, batch_size=batch_size, shuffle=False)

    return test_loader, test_window_loader, test_question_loader, test_question_window_loader

def update_gap(max_rgap, max_sgap, max_pcount, cur):
    max_rgap = cur.max_rgap if cur.max_rgap > max_rgap else max_rgap
    max_sgap = cur.max_sgap if cur.max_sgap > max_sgap else max_sgap
    max_pcount = cur.max_pcount if cur.max_pcount > max_pcount else max_pcount
    return max_rgap, max_sgap, max_pcount

def init_dataset4train(dataset_name, model_name, data_config, i, batch_size, diff_level=None, args=None, not_select_dataset=None, re_mapping=False):
    print(f"dataset_name:{dataset_name}")
    print(f"data_config:{data_config}")
    data_config = data_config[dataset_name]
    all_folds = set(data_config["folds"])
    if model_name in ["dkt_forget", "bakt_time", "fa_kt", "mtkt"]:
        max_rgap, max_sgap, max_pcount = 0, 0, 0
        curvalid = DktForgetDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], {i})
        curtrain = DktForgetDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], all_folds - {i})
        max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, curtrain)
        max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, curvalid)
    elif model_name == "lpkt":
        at2idx, it2idx = generate_time2idx(data_config)
        # json_str = json.dumps(at2idx)
        # with open('at2idx.json', 'w') as json_file:
        #     json_file.write(json_str)
        # json_str_2 = json.dumps(it2idx)
        # with open('it2idx.json', 'w') as json_file2:
        #     json_file2.write(json_str_2)
        curvalid = LPKTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]), at2idx, it2idx, data_config["input_type"], {i})
        curtrain = LPKTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]), at2idx, it2idx, data_config["input_type"], all_folds - {i})
    elif model_name in ["rkt"] and dataset_name in ["statics2011", "assist2015", "poj"]:
        curvalid = KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], {i})
        curtrain = KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], all_folds - {i})
    elif model_name in que_type_models:
        if model_name in ["promptkt"]:
            dataset_name = args.dataset_name
            train_ratio = args.dataset_name
            if args.train_mode == "pretrain":
                dpath = os.path.join(
                    data_config["dpath"],
                    f"train_valid_sequences_quelevel_pretrain_nomapping.csv",
                )
            else:
                dpath = os.path.join(
                    data_config["dpath"],
                    f"train_valid_sequences_quelevel.csv",
                )
            print(f"train_data_path:{dpath}")
            if not os.path.exists(dpath) and args.train_mode == "pretrain":
                print(f"loading pretrain data")

                get_pretrain_data(data_config)
            curvalid = KTQueDataset_promptKT(
                dpath,
                input_type=data_config["input_type"],
                folds={i},
                concept_num=data_config["num_c"],
                max_concepts=data_config["max_concepts"],
                not_select_dataset=not_select_dataset,
                train_ratio=train_ratio,
                dataset_name=dataset_name,
            )
            curtrain = KTQueDataset_promptKT(
                dpath,
                input_type=data_config["input_type"],
                folds=all_folds - {i},
                concept_num=data_config["num_c"],
                max_concepts=data_config["max_concepts"],
                not_select_dataset=not_select_dataset,
                train_ratio=train_ratio,
                dataset_name=dataset_name,
            )
        else:
            curvalid = KTQueDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                            input_type=data_config["input_type"], folds={i}, 
                            concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
            curtrain = KTQueDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                            input_type=data_config["input_type"], folds=all_folds - {i}, 
                            concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
    elif model_name in ["atdkt"]:
        curvalid = ATDKTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], {i})
        curtrain = ATDKTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], all_folds - {i})
    elif model_name == "dimkt":
        curvalid = DIMKTDataset(data_config["dpath"],os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], {i}, diff_level=diff_level)
        curtrain = DIMKTDataset(data_config["dpath"],os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], all_folds - {i}, diff_level=diff_level)
    elif model_name == "mockt":
        curvalid = MOCKTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], {i})
        curtrain = MOCKTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], all_folds - {i})
    elif model_name in NATIVE_EVIDENCE_MODELS:
        path = os.path.join(
            data_config["dpath"], data_config["train_valid_file"]
        )
        num_items = max(1, int(data_config["num_q"]) + 1)
        num_concepts = int(data_config["num_c"]) + 1
        dataset_cls = R291Dataset if model_name in ["r291_conserved_evidence", "r292_budgeted_schema", "r293_evidence_retrieval", "r294_protected_retrieval", "r295_source_evidence", "r295_temporal_evidence", "r295_role_temporal", "r296_source_owned", "r297_contextual_sources", "r298_centered_sources", "r299_factored_sources", "r300_temporal_sources", "r301_aligned_moments"] else R239Dataset
        if model_name in ("r303_crossfit_reference", "r305_path_ablations", "r307_grounded_history", "r309_write_separation", "r310_transfer_supervision", "r311_role_ablation", "r312_role_separation"):
            dataset_cls = R303Dataset
        curtrain = dataset_cls(
            path,
            data_config["input_type"],
            all_folds - {i},
            num_items,
            num_concepts,
        )
        data_config["r239_priors"] = curtrain.priors
        if model_name in ("r303_crossfit_reference", "r305_path_ablations", "r307_grounded_history", "r309_write_separation", "r310_transfer_supervision", "r311_role_ablation", "r312_role_separation"):
            data_config["r303_reference_audit"] = curtrain.reference_audit
        curvalid = dataset_cls(
            path,
            data_config["input_type"],
            {i},
            num_items,
            num_concepts,
            priors=curtrain.priors,
        )
    else:
        curvalid = KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], {i})
        curtrain = KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], all_folds - {i})

    if model_name in ("asikt", "asikt_official"):
        data_config["asikt_priors"] = build_asikt_priors(
            curtrain,
            data_config["num_q"],
            data_config["num_c"],
        )

    train_loader = DataLoader(
        curtrain,
        batch_size=batch_size,
        shuffle=(
            model_name in NATIVE_EVIDENCE_MODELS
            or model_name in ("asikt", "asikt_official")
        ),
    )
    valid_loader = DataLoader(curvalid, batch_size=batch_size)
    
    try:
        if model_name in ["dkt_forget", "bakt_time", "fa_kt", "mtkt"]:
            test_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
            # test_window_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]),
            #                                 data_config["input_type"], {-1})
            max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, test_dataset)
    #     elif model_name == "lpkt":
    #         test_dataset = LPKTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), at2idx, it2idx, data_config["input_type"], {-1})
    #         # test_window_dataset = LPKTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]), at2idx, it2idx, data_config["input_type"], {-1})
    #     elif model_name in que_type_models:
    #         test_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
    #                         input_type=data_config["input_type"], folds=[-1], 
    #                         concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
    #     else:
    #         test_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
    #         # test_window_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]), data_config["input_type"], {-1})
    except:
        pass
    
    if model_name in ["dkt_forget", "bakt_time", "fa_kt", "mtkt"]:
        data_config["num_rgap"] = max_rgap + 1
        data_config["num_sgap"] = max_sgap + 1
        data_config["num_pcount"] = max_pcount + 1
    if model_name == "lpkt":
        print(f"num_at:{len(at2idx)}")
        print(f"num_it:{len(it2idx)}")
        data_config["num_at"] = len(at2idx) + 1
        data_config["num_it"] = len(it2idx) + 1
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # # test_window_loader = DataLoader(test_window_dataset, batch_size=batch_size, shuffle=False)
    # test_window_loader = None
    return train_loader, valid_loader#, test_loader, test_window_loader
