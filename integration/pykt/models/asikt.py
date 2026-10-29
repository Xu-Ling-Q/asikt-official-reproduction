import math

import torch
from torch import nn

from .asikt_mamba2 import Mamba2, Mamba2Config


def _local_accuracy(ids, answers, valid, bayes_accuracy, frequency):
    length = ids.shape[1]
    same_id = ids.unsqueeze(2).eq(ids.unsqueeze(1))
    update_mask = torch.tril(
        torch.ones(
            length, length, dtype=torch.bool, device=ids.device
        )
    )
    update_mask[:, 0] = False
    update_mask = (
        update_mask.unsqueeze(0)
        & valid.unsqueeze(1)
        & valid.unsqueeze(2)
        & same_id
    )
    previous_answers = torch.cat(
        [torch.zeros_like(answers[:, :1]), answers[:, :-1]], dim=1
    )
    update_values = update_mask.to(answers.dtype)
    local_count = update_values.sum(dim=-1)
    local_correct = (
        update_values * previous_answers.unsqueeze(1)
    ).sum(dim=-1)

    global_accuracy = bayes_accuracy[ids]
    global_frequency = frequency[ids]
    accuracy = (
        local_correct + global_frequency * global_accuracy
    ) / (local_count + global_frequency).clamp_min(1e-8)
    return torch.where(valid, accuracy, torch.zeros_like(accuracy))


def build_difficulty_masks(
    question_ids,
    concept_ids,
    answers,
    valid,
    question_bacc,
    question_frequency,
    concept_bacc,
    concept_frequency,
    theta=0.5,
):
    """Build the paper's Exercise Weight Matrix without storing N x L x L."""
    question_accuracy = _local_accuracy(
        question_ids,
        answers,
        valid,
        question_bacc,
        question_frequency,
    )
    concept_accuracy = _local_accuracy(
        concept_ids,
        answers,
        valid,
        concept_bacc,
        concept_frequency,
    )

    question_weight = (1.0 - question_accuracy) * valid
    pair_weight = question_weight.unsqueeze(2) * question_weight.unsqueeze(1)
    same_concept = concept_ids.unsqueeze(2).eq(concept_ids.unsqueeze(1))
    same_answer = answers.unsqueeze(2).eq(answers.unsqueeze(1))
    concept_factor = torch.exp(
        1.0 - concept_accuracy
    ).unsqueeze(1)

    weighted_same_concept = pair_weight * concept_factor
    question_mask = torch.where(
        same_concept, weighted_same_concept, pair_weight
    )
    answer_factor = torch.where(
        same_answer,
        torch.ones_like(pair_weight),
        torch.full_like(pair_weight, float(theta)),
    )
    interaction_mask = torch.where(
        same_concept,
        weighted_same_concept,
        pair_weight * answer_factor,
    )

    length = question_ids.shape[1]
    causal = torch.tril(
        torch.ones(
            length, length, dtype=torch.bool, device=question_ids.device
        )
    )
    pair_valid = (
        causal.unsqueeze(0)
        & valid.unsqueeze(1)
        & valid.unsqueeze(2)
    )
    zeros = torch.zeros_like(question_mask)
    return (
        torch.where(pair_valid, question_mask, zeros),
        torch.where(pair_valid, interaction_mask, zeros),
    )


class ASIKT(nn.Module):
    """pyKT adapter for the official ASIKT paper implementation."""

    def __init__(
        self,
        num_concepts,
        num_questions,
        priors,
        d_model=128,
        dropout=0.2,
        n_blocks=2,
        rasch=1,
        final_fc_dim=512,
        n_heads=8,
        d_ff=1024,
        theta=0.5,
        mamba_chunk_size=10,
        gradient_accumulation_steps=1,
        emb_type="qid",
    ):
        super().__init__()
        self.model_name = "asikt"
        self.emb_type = emb_type
        self.num_c = int(num_concepts)
        self.num_q = int(num_questions)
        self.rasch = bool(rasch)
        self.theta = float(theta)
        self.gradient_accumulation_steps = max(
            1, int(gradient_accumulation_steps)
        )

        self.q_embed = nn.Embedding(self.num_q + 1, d_model)
        self.kc_embed = nn.Embedding(self.num_c + 1, d_model)
        self.a_embed = nn.Embedding(2, d_model)
        if self.rasch:
            self.deviates_para = nn.Embedding(self.num_q + 1, 1)
            self.kc_embed_deviates = nn.Embedding(
                self.num_c + 1, d_model
            )
            self.a_embed_deviates = nn.Embedding(2, d_model)

        self.architecture = ASIKTArchitecture(
            n_blocks=n_blocks,
            n_heads=n_heads,
            dropout=dropout,
            d_model=d_model,
            d_ff=d_ff,
            mamba_chunk_size=mamba_chunk_size,
        )
        self.out = self._prediction_head(
            d_model * 2, final_fc_dim, dropout
        )
        self.slipping = self._prediction_head(
            d_model * 2, final_fc_dim, dropout
        )
        self.guessing = self._prediction_head(
            d_model * 2, final_fc_dim, dropout
        )

        self.register_buffer(
            "question_bacc", priors["question_bacc"].float()
        )
        self.register_buffer(
            "question_frequency",
            priors["question_frequency"].float(),
        )
        self.register_buffer(
            "concept_bacc", priors["concept_bacc"].float()
        )
        self.register_buffer(
            "concept_frequency", priors["concept_frequency"].float()
        )
        self.reset()

    @staticmethod
    def _prediction_head(input_size, hidden_size, dropout):
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def reset(self):
        # The released ASIKT source initializes the question table at zero.
        nn.init.zeros_(self.q_embed.weight)
        if self.rasch:
            nn.init.zeros_(self.deviates_para.weight)

    def _prepare_sequences(self, data):
        device = self.q_embed.weight.device
        masks = data["masks"].to(device).bool()
        questions = torch.cat(
            [
                data["qseqs"][:, :1],
                data["shft_qseqs"],
            ],
            dim=1,
        ).to(device)
        concepts = torch.cat(
            [
                data["cseqs"][:, :1],
                data["shft_cseqs"],
            ],
            dim=1,
        ).to(device)
        answers = torch.cat(
            [
                data["rseqs"][:, :1],
                data["shft_rseqs"],
            ],
            dim=1,
        ).to(device)
        valid = torch.cat([masks[:, :1], masks], dim=1)

        questions = torch.where(valid, questions.long() + 1, 0)
        concepts = torch.where(valid, concepts.long() + 1, 0)
        answers = torch.where(valid, answers.long(), 0)
        return questions, concepts, answers, valid

    def forward(self, data):
        questions, concepts, answers, valid = self._prepare_sequences(
            data
        )
        question_mask, interaction_mask = build_difficulty_masks(
            questions,
            concepts,
            answers,
            valid,
            self.question_bacc,
            self.question_frequency,
            self.concept_bacc,
            self.concept_frequency,
            self.theta,
        )

        q_embed = self.q_embed(questions)
        kc_embed = self.kc_embed(concepts)
        a_embed = self.a_embed(answers)
        kca_embed = kc_embed + a_embed
        qa_embed = q_embed + a_embed

        if self.rasch:
            q_deviation = self.deviates_para(questions)
            kc_deviation = self.kc_embed_deviates(concepts)
            a_deviation = self.a_embed_deviates(answers)
            q_embed = kc_embed + q_deviation * kc_deviation
            qa_embed = (
                kca_embed
                + q_deviation * (a_deviation + kc_deviation)
            )

        state = self.architecture(
            q_embed,
            kc_embed,
            qa_embed,
            kca_embed,
            question_mask,
            interaction_mask,
        )
        prediction_input = torch.cat([state, kc_embed], dim=-1)
        state_output = self.out(prediction_input)
        slipping = self.slipping(prediction_input)
        guessing = self.guessing(prediction_input)
        logits = (
            (1.0 - state_output) * guessing
            + state_output * (1.0 - slipping)
        )
        return torch.sigmoid(logits.squeeze(-1))[:, :-1]


class ASIKTArchitecture(nn.Module):
    def __init__(
        self,
        n_blocks,
        n_heads,
        dropout,
        d_model,
        d_ff,
        mamba_chunk_size,
    ):
        super().__init__()
        self.q_blocks = self._blocks(
            n_blocks, d_model, d_ff, n_heads, dropout
        )
        self.qa_blocks = self._blocks(
            n_blocks, d_model, d_ff, n_heads, dropout
        )
        self.kc_blocks = self._blocks(
            n_blocks, d_model, d_ff, n_heads, dropout
        )
        self.kca_blocks = self._blocks(
            n_blocks, d_model, d_ff, n_heads, dropout
        )
        self.retriever = Mamba2(
            Mamba2Config(
                d_model=d_model * 4,
                chunk_size=mamba_chunk_size,
            )
        )
        self.linear = nn.Linear(d_model * 4, d_model)
        self.selector = Mamba2(
            Mamba2Config(
                d_model=d_model,
                chunk_size=mamba_chunk_size,
            )
        )

    @staticmethod
    def _blocks(n_blocks, d_model, d_ff, n_heads, dropout):
        return nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    d_feature=d_model // n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_heads=n_heads,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(
        self,
        q_embed,
        kc_embed,
        qa_embed,
        kca_embed,
        question_mask,
        interaction_mask,
    ):
        q_state = q_embed
        kc_state = kc_embed
        qa_state = qa_embed
        kca_state = kca_embed
        for block in self.q_blocks:
            q_state = block(
                q_state, q_state, q_state, question_mask
            )
        for block in self.qa_blocks:
            qa_state = block(
                qa_state, qa_state, qa_state, interaction_mask
            )
        for block in self.kc_blocks:
            kc_state = block(
                kc_state, kc_state, kc_state, question_mask
            )
        for block in self.kca_blocks:
            kca_state = block(
                kca_state, kca_state, kca_state, interaction_mask
            )

        q_next = torch.cat(
            [q_state[:, 1:], torch.zeros_like(q_state[:, :1])],
            dim=1,
        )
        kc_next = torch.cat(
            [kc_state[:, 1:], torch.zeros_like(kc_state[:, :1])],
            dim=1,
        )
        retrieval_input = torch.cat(
            [qa_state, kca_state, q_next, kc_next], dim=-1
        )
        retrieved, _ = self.retriever(retrieval_input)
        selected, _ = self.selector(torch.sigmoid(self.linear(retrieved)))
        return selected


class TransformerLayer(nn.Module):
    def __init__(
        self, d_model, d_feature, d_ff, n_heads, dropout
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key, values, difficulty_mask):
        length = query.size(1)
        causal = torch.tril(
            torch.ones(
                length,
                length,
                dtype=torch.bool,
                device=query.device,
            )
        )
        attended = self.attention(
            query,
            key,
            values,
            difficulty_mask,
            causal.unsqueeze(0).unsqueeze(0),
        )
        query = self.norm1(query + self.dropout1(attended))
        feed_forward = self.linear2(
            self.dropout(self.activation(self.linear1(query)))
        )
        return self.norm2(query + self.dropout2(feed_forward))


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_model, d_feature, n_heads, dropout, bias=True
    ):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.n_heads = n_heads
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.memory_floor = nn.Parameter(
            torch.zeros(n_heads, 1, 1)
        )
        self.memory_decay = nn.Parameter(
            torch.zeros(n_heads, 1, 1)
        )
        nn.init.xavier_uniform_(self.memory_floor)
        nn.init.xavier_uniform_(self.memory_decay)

    def forward(
        self, query, key, values, difficulty_mask, causal_mask
    ):
        batch_size, length = query.shape[:2]
        key = self.k_linear(key).view(
            batch_size, length, self.n_heads, self.d_k
        )
        query = self.k_linear(query).view(
            batch_size, length, self.n_heads, self.d_k
        )
        values = self.v_linear(values).view(
            batch_size, length, self.n_heads, self.d_k
        )
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(self.d_k)
        positions = torch.arange(length, device=query.device)
        distance = torch.abs(
            positions.unsqueeze(0) - positions.unsqueeze(1)
        ).float()
        floor = torch.sigmoid(self.memory_floor).unsqueeze(0)
        decay = torch.nn.functional.softplus(
            self.memory_decay
        ).unsqueeze(0)
        memory = floor + (1.0 - floor) * torch.exp(
            -decay * distance.unsqueeze(0).unsqueeze(0)
        )
        scores = (
            scores
            * difficulty_mask.unsqueeze(1)
            * memory
        )
        scores = scores.masked_fill(~causal_mask, -1e32)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(weights, values)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, length, self.d_model
        )
        return self.out_proj(output)
