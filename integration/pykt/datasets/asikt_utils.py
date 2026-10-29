import torch


def _smoothed_accuracy(ids, responses, valid, size):
    ids = ids[valid].long() + 1
    responses = responses[valid].float()

    counts = torch.bincount(ids, minlength=size).float()
    correct = torch.bincount(ids, weights=responses, minlength=size).float()
    seen = counts > 0
    seen[0] = False
    if not torch.any(seen):
        raise ValueError("ASIKT priors require at least one valid interaction.")

    raw_accuracy = correct[seen] / counts[seen]
    average_count = counts[seen].sum() / seen.sum()
    average_accuracy = raw_accuracy.mean()

    frequency = torch.full((size,), average_count, dtype=torch.float32)
    bayes_accuracy = torch.full(
        (size,), average_accuracy, dtype=torch.float32
    )
    frequency[seen] = counts[seen]
    bayes_accuracy[seen] = (
        correct[seen] + average_count * average_accuracy
    ) / (counts[seen] + average_count)
    frequency[0] = 0.0
    bayes_accuracy[0] = 0.0
    return bayes_accuracy, frequency


def build_asikt_priors(dataset, num_questions, num_concepts):
    """Estimate ASIKT difficulty priors from one training-only dataset."""
    data = dataset.dori
    if len(data.get("qseqs", [])) == 0 or len(data.get("cseqs", [])) == 0:
        raise ValueError("ASIKT requires both question and concept sequences.")

    questions = data["qseqs"].cpu()
    concepts = data["cseqs"].cpu()
    responses = data["rseqs"].cpu()
    valid = (questions >= 0) & (concepts >= 0)

    q_bacc, q_freq = _smoothed_accuracy(
        questions, responses, valid, int(num_questions) + 1
    )
    c_bacc, c_freq = _smoothed_accuracy(
        concepts, responses, valid, int(num_concepts) + 1
    )
    return {
        "question_bacc": q_bacc,
        "question_frequency": q_freq,
        "concept_bacc": c_bacc,
        "concept_frequency": c_freq,
    }
