import torch
import numpy as np
import os

from .dkt import DKT
from .dkt_plus import DKTPlus
from .dkvmn import DKVMN
from .deep_irt import DeepIRT
from .sakt import SAKT
from .saint import SAINT
from .kqn import KQN
from .atkt import ATKT
from .dkt_forget import DKTForget
from .akt import AKT
from .gkt import GKT
from .gkt_utils import get_gkt_graph
from .lpkt import LPKT
from .lpkt_utils import generate_qmatrix
from .skvmn import SKVMN
from .hawkes import HawkesKT
from .iekt import IEKT
from .atdkt import ATDKT
from .simplekt import simpleKT
from .datakt import BAKTTime
from .qdkt import QDKT
from .qikt import QIKT
from .dimkt import DIMKT
from .sparsekt import sparseKT
from .rkt import RKT
from .folibikt import folibiKT
from .r239_evidence_schema import R239EvidenceSchemaKT
from .r290_rank_aligned import R290RankAlignedKT
from .r291_conserved_evidence import R291ConservedEvidenceKT
from .r292_budgeted_schema import R292BudgetedSchemaKT
from .r293_evidence_retrieval import R293EvidenceRetrievalKT
from .r294_protected_retrieval import R294ProtectedRetrievalKT
from .r295_source_evidence import R295SourceEvidenceKT
from .r295_temporal_evidence import R295TemporalEvidenceKT
from .r295_role_temporal import R295RoleTemporalKT
from .r296_source_owned import R296SourceOwnedKT
from .r297_contextual_sources import R297ContextualSourcesKT
from .r298_centered_sources import R298CenteredSourcesKT
from .r299_factored_sources import R299FactoredSourcesKT
from .r300_temporal_sources import R300TemporalSourcesKT
from .r301_aligned_moments import R301AlignedMomentsKT
from .r303_crossfit_reference import R303CrossfitReferenceKT
from .r305_path_ablations import R305PathAblationsKT
from .r307_grounded_history import R307GroundedHistoryKT
from .r309_write_separation import R309WriteSeparationKT
from .r310_transfer_supervision import R310TransferSupervisionKT
from .r311_role_ablation import R311RoleAblationKT
from .r312_role_separation import R312RoleSeparationKT
from .r313_orthogonal_residual import R313OrthogonalResidualKT
from .r314_selective_residual import R314SelectiveResidualKT
from .r316_transfer_ledger import R316TransferLedgerKT
from .r265_dual_timescale import R265DualTimescaleKT
from .dtransformer import DTransformer
from .stablekt import stableKT
from .extrakt import extraKT
from .rekt import ReKT
from .cskt import CSKT
from .lefokt_akt import LEFOKT_AKT
from .ukt import UKT
from .hcgkt import HCGKT
from .robustkt import Robustkt
from .mockt import MocKT
from .fa_kt import FA_KT
from .mtkt import MTKT
from .denoisekt import DenoiseKT
from .fluckt import FlucKT
from .cgmkt import CGMKT
from .dkt_enhance_pro import DKT_Enhance_Pro
from .dkvmn_enhance_pro import DKVMN_Enhance_PRO
from .sakt_enhance_pro import SAKT_Enhance_PRO
from .akt_enhance_pro_qid import AKT_Enhance_Pro_qid
from .simplekt_enhance_pro_qid import simpleKT_enhance_pro_qid
from .asikt import ASIKT
from .asikt_official import ASIKTOfficial

device = "cpu" if not torch.cuda.is_available() else "cuda"

def init_model(model_name, model_config, data_config, emb_type, dataset_name=None):
    if dataset_name is None:
        dataset_name = data_config.get("dataset_name")
    if dataset_name is None and data_config.get("dpath"):
        dataset_name = os.path.basename(os.path.normpath(data_config["dpath"]))
    if model_name == "dkt":
        model = DKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkt+":
        model = DKTPlus(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkvmn":
        model = DKVMN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "deep_irt":
        model = DeepIRT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "sakt":
        model = SAKT(data_config["num_c"],  **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "saint":
        model = SAINT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkt_forget":
        model = DKTForget(data_config["num_c"], data_config["num_rgap"], data_config["num_sgap"], data_config["num_pcount"], **model_config).to(device)
    elif model_name == "akt":
        model = AKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "lefokt_akt":
        model = LEFOKT_AKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "extrakt":
        model = extraKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "folibikt":
        model = folibiKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "kqn":
        model = KQN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "atkt":
        model = ATKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], fix=False).to(device)
    elif model_name == "atktfix":
        model = ATKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], fix=True).to(device)
    elif model_name == "gkt":
        graph_type = model_config['graph_type']
        fname = f"gkt_graph_{graph_type}.npz"
        graph_path = os.path.join(data_config["dpath"], fname)
        if os.path.exists(graph_path):
            graph = torch.tensor(np.load(graph_path, allow_pickle=True)['matrix']).float()
        else:
            graph = get_gkt_graph(data_config["num_c"], data_config["dpath"], 
                    data_config["train_valid_original_file"], data_config["test_original_file"], graph_type=graph_type, tofile=fname)
            graph = torch.tensor(graph).float()
        model = GKT(data_config["num_c"], **model_config,graph=graph,emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "lpkt":
        qmatrix_path = os.path.join(data_config["dpath"], "qmatrix.npz")
        if os.path.exists(qmatrix_path):
            q_matrix = np.load(qmatrix_path, allow_pickle=True)['matrix']
        else:
            q_matrix = generate_qmatrix(data_config)
        q_matrix = torch.tensor(q_matrix).float().to(device)
        model = LPKT(data_config["num_at"], data_config["num_it"], data_config["num_q"], data_config["num_c"], **model_config, q_matrix=q_matrix, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "skvmn":
        model = SKVMN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)   
    elif model_name == "hawkes":
        if data_config["num_q"] == 0 or data_config["num_c"] == 0:
            print(f"model: {model_name} needs questions ans concepts! but the dataset has no both")
            return None
        model = HawkesKT(data_config["num_c"], data_config["num_q"], **model_config)
        model = model.double()
        # print("===before init weights"+"@"*100)
        # model.printparams()
        model.apply(model.init_weights)
        # print("===after init weights")
        # model.printparams()
        model = model.to(device)
    elif model_name == "iekt":
        model = IEKT(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)   
    elif model_name == "qdkt":
        model = QDKT(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)
    elif model_name == "qikt":
        model = QIKT(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)
    elif model_name == "atdkt":
        model = ATDKT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "datakt":
        model = BAKTTime(data_config["num_c"], data_config["num_q"], data_config["num_rgap"], data_config["num_sgap"], data_config["num_pcount"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "simplekt":
        model = simpleKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "rekt":
        model = ReKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type).to(device)
    elif model_name == "stablekt":
        model = stableKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dimkt":
        model = DIMKT(data_config["num_q"],data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "sparsekt":
        model = sparseKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "rkt":
        model = RKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device) 
    elif model_name == "cskt":
        model = CSKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device) 
    elif model_name == "ukt":
        model = UKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "hcgkt":
        model = HCGKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "robustkt":
        model = Robustkt(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dtransformer":
        model = DTransformer(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type,
                     emb_path=data_config["emb_path"]).to(device)
    elif model_name == "mockt":
        model = MocKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "fa_kt":
        import inspect
        fa_kt_params = inspect.signature(FA_KT.__init__).parameters
        fa_kt_config = {k: v for k, v in model_config.items() if k in fa_kt_params}
        model = FA_KT(data_config["num_c"], data_config["num_q"], data_config["num_rgap"], data_config["num_sgap"], data_config["num_pcount"], **fa_kt_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "mtkt":
        model = MTKT(data_config["num_c"], data_config["num_q"], data_config["num_rgap"], data_config["num_sgap"], data_config["num_pcount"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "denoisekt":
        model = DenoiseKT(num_c=data_config['num_c'], num_q=data_config['num_q'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, dpath=data_config["dpath"], emb_path=data_config["emb_path"], device=device).to(device)
    elif model_name == "fluckt":
        model = FlucKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkt_enhance_pro":
        model = DKT_Enhance_Pro(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], dataset_name=dataset_name).to(device)
    elif model_name == "dkvmn_enhance_pro":
        model = DKVMN_Enhance_PRO(data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], dataset_name=dataset_name).to(device)
    elif model_name == "sakt_enhance_pro":
        model = SAKT_Enhance_PRO(data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], dataset_name=dataset_name).to(device)
    elif model_name == "akt_enhance_pro_qid":
        model = AKT_Enhance_Pro_qid(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], dataset_name=dataset_name).to(device)
    elif model_name == "simplekt_enhance_pro_qid":
        model = simpleKT_enhance_pro_qid(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], dataset_name=dataset_name).to(device)
    elif model_name == "asikt":
        if "asikt_priors" not in data_config:
            raise RuntimeError(
                "ASIKT priors are missing. Initialize the native dataset first "
                "with init_dataset4train so only training folds estimate priors."
            )
        model = ASIKT(
            data_config["num_c"],
            data_config["num_q"],
            data_config["asikt_priors"],
            **model_config,
            emb_type=emb_type,
        ).to(device)
    elif model_name == "asikt_official":
        if "asikt_priors" not in data_config:
            raise RuntimeError(
                "ASIKT priors are missing. Initialize the native dataset first "
                "with init_dataset4train so only training folds estimate priors."
            )
        model = ASIKTOfficial(
            data_config["num_c"],
            data_config["num_q"],
            data_config["asikt_priors"],
            **model_config,
            emb_type=emb_type,
        ).to(device)
    elif model_name == "cgmkt":
        model = CGMKT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], dataset_name=dataset_name).to(device)
    elif model_name == "cdfkt":
        from .cdfkt import CDFKT
        model = CDFKT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], dataset_name=dataset_name).to(device)
    elif model_name in ["r239_evidence_schema", "r265_dual_timescale", "r290_rank_aligned", "r291_conserved_evidence", "r292_budgeted_schema", "r293_evidence_retrieval", "r294_protected_retrieval", "r295_source_evidence", "r295_temporal_evidence", "r295_role_temporal", "r296_source_owned", "r297_contextual_sources", "r298_centered_sources", "r299_factored_sources", "r300_temporal_sources", "r301_aligned_moments", "r303_crossfit_reference", "r305_path_ablations", "r307_grounded_history", "r309_write_separation", "r310_transfer_supervision", "r311_role_ablation", "r312_role_separation", "r313_orthogonal_residual", "r314_selective_residual", "r316_transfer_ledger"]:
        if "r239_priors" not in data_config:
            raise RuntimeError(
                "R239 priors are missing. Initialize the native dataset first "
                "with init_dataset4train so only training folds estimate priors."
            )
        model_cls = {
            "r239_evidence_schema": R239EvidenceSchemaKT,
            "r265_dual_timescale": R265DualTimescaleKT,
            "r290_rank_aligned": R290RankAlignedKT,
            "r291_conserved_evidence": R291ConservedEvidenceKT,
            "r292_budgeted_schema": R292BudgetedSchemaKT,
            "r293_evidence_retrieval": R293EvidenceRetrievalKT,
            "r294_protected_retrieval": R294ProtectedRetrievalKT,
            "r295_source_evidence": R295SourceEvidenceKT,
            "r295_temporal_evidence": R295TemporalEvidenceKT,
            "r295_role_temporal": R295RoleTemporalKT,
            "r296_source_owned": R296SourceOwnedKT,
            "r297_contextual_sources": R297ContextualSourcesKT,
            "r298_centered_sources": R298CenteredSourcesKT,
            "r299_factored_sources": R299FactoredSourcesKT,
            "r300_temporal_sources": R300TemporalSourcesKT,
            "r301_aligned_moments": R301AlignedMomentsKT,
            "r303_crossfit_reference": R303CrossfitReferenceKT,
            "r305_path_ablations": R305PathAblationsKT,
    "r307_grounded_history": R307GroundedHistoryKT,
            "r309_write_separation": R309WriteSeparationKT,
            "r310_transfer_supervision": R310TransferSupervisionKT,
            "r311_role_ablation": R311RoleAblationKT,
            "r312_role_separation": R312RoleSeparationKT,
            "r313_orthogonal_residual": R313OrthogonalResidualKT,
            "r314_selective_residual": R314SelectiveResidualKT,
            "r316_transfer_ledger": R316TransferLedgerKT,
        }[model_name]
        model = model_cls(
            data_config["num_q"],
            data_config["num_c"],
            data_config["r239_priors"],
            initialization_file=(
                None if model_name in ["r291_conserved_evidence", "r292_budgeted_schema", "r293_evidence_retrieval", "r294_protected_retrieval", "r295_source_evidence", "r295_temporal_evidence", "r295_role_temporal", "r296_source_owned", "r297_contextual_sources", "r298_centered_sources", "r299_factored_sources", "r300_temporal_sources", "r301_aligned_moments", "r303_crossfit_reference", "r305_path_ablations", "r307_grounded_history", "r309_write_separation", "r310_transfer_supervision", "r311_role_ablation", "r312_role_separation", "r313_orthogonal_residual", "r314_selective_residual", "r316_transfer_ledger"]
                else data_config.get("r239_initialization_file")
            ),
            **model_config,
            emb_type=emb_type,
        ).to(device)
    else:
        print("The wrong model name was used...")
        return None
    return model

def load_model(model_name, model_config, data_config, emb_type, ckpt_path, dataset_name=None):
    model = init_model(model_name, model_config, data_config, emb_type, dataset_name)
    net = torch.load(os.path.join(ckpt_path, emb_type+"_model.ckpt"))
    model.load_state_dict(net)
    return model
