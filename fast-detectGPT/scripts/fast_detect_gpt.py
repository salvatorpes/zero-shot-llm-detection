# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import argparse
import json
from data_builder import load_data
from model import load_tokenizer, load_model, openai_models
from metrics import get_roc_metrics, get_precision_recall_metrics

def get_samples(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    nsamples = 10000
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood.mean(dim=1)

# this is the actual fastdetectGPT
def get_sampling_discrepancy(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    samples = get_samples(logits_ref, labels)
    log_likelihood_x = get_likelihood(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples)
    miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
    discrepancy = (log_likelihood_x.squeeze(-1) - miu_tilde) / sigma_tilde
    return discrepancy.item()

def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    discrepancy = discrepancy.mean()
    return discrepancy.item()

def _approx_moments_from_tops(ref_top_list, score_top_list, eps: float = 1e-8):
    """
    ref_top_list: list[{'token', 'logprob', 'rank'}] for a single position (from ref model)
    score_top_list: list[{'token', 'logprob', 'rank'}] for a single position (from scoring model)
    Returns mean_ref, var_ref using the intersection of tokens and renormalized ref probs.
    """
    if not ref_top_list or not score_top_list:
        return 0.0, eps  # avoid zero variance

    ref_map = {d["token"]: float(d["logprob"]) for d in ref_top_list}
    score_map = {d["token"]: float(d["logprob"]) for d in score_top_list}
    keys = list(set(ref_map.keys()) & set(score_map.keys()))
    if not keys:
        return 0.0, eps

    ref_lp = np.array([ref_map[k] for k in keys], dtype=np.float64)
    # normalize ref to probabilities over the intersection (truncated softmax)
    ref_p = np.exp(ref_lp - ref_lp.max())
    s = ref_p.sum()
    if s <= 0:
        return 0.0, eps
    ref_p = ref_p / s

    score_lp = np.array([score_map[k] for k in keys], dtype=np.float64)
    mean = float(np.sum(ref_p * score_lp))
    second = float(np.sum(ref_p * np.square(score_lp)))
    var = max(second - mean * mean, eps)
    return mean, var

def openai_discrepancy_analytic_from_logs(score_logs, ref_logs, eps: float = 1e-8) -> float:
    """
    score_logs/ref_logs are dicts from OpenAIModel.score_text_with_logprobs(separate=True).
    Mirrors get_sampling_discrepancy_analytic using top-logprobs intersection.
    """
    # log-likelihood of the *observed* tokens under scoring model
    token_ll = np.array(score_logs["token_logprobs"], dtype=np.float64)  # one per position

    # per-position approximate moments under reference distribution
    means, vars_ = [], []
    for t in range(len(token_ll)):
        ref_top = ref_logs["top_logprobs_per_token"][t] if t < len(ref_logs["top_logprobs_per_token"]) else []
        score_top = score_logs["top_logprobs_per_token"][t] if t < len(score_logs["top_logprobs_per_token"]) else []
        m, v = _approx_moments_from_tops(ref_top, score_top, eps=eps)
        means.append(m)
        vars_.append(v)

    means = np.array(means, dtype=np.float64)
    vars_ = np.array(vars_, dtype=np.float64)
    num = float(np.sum(token_ll) - np.sum(means))
    den = float(np.sqrt(np.sum(vars_) + eps))
    return num / den if den > 0 else 0.0


def experiment(args):

    is_openai_model = args.scoring_model_name in openai_models

    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()

    if args.sampling_model_name != args.scoring_model_name:
        sampling_tokenizer = load_tokenizer(args.sampling_model_name, args.cache_dir)
        sampling_model = load_model(args.sampling_model_name, args.device, args.cache_dir)
        sampling_model.eval()
            
    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])
    
    # evaluate criterion
    if args.discrepancy_analytic:
        name = "sampling_discrepancy_analytic"
        criterion_fn = get_sampling_discrepancy_analytic
    else:
        name = "sampling_discrepancy"
        criterion_fn = get_sampling_discrepancy

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    results = []
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        
        if is_openai_model:
            top_k, max_length = 20, 300

            sampled_score_logs = data["sampled_logprobs"][idx]
            original_score_logs = scoring_model.score_text_with_logprobs(text=original_text, top_k=top_k, max_length=300)

            if args.sampling_model_name == args.scoring_model_name:
                original_ref_logs = original_score_logs
                sampled_ref_logs = sampled_score_logs
            else:
                original_ref_logs = sampling_model.score_text_with_logprobs(text=original_text, top_k=top_k, max_length=300)
                sampled_ref_logs = sampling_model.score_text_with_logprobs(text=sampled_text, top_k=top_k, max_length=300)

            original_crit = openai_discrepancy_analytic_from_logs(original_score_logs, original_ref_logs, eps=1e-8)
            sampled_crit = openai_discrepancy_analytic_from_logs(sampled_score_logs, sampled_ref_logs, eps=1e-8)

        else:
            # original text
            tokenized = scoring_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
            labels = tokenized.input_ids[:, 1:]
            with torch.no_grad():
                logits_score = scoring_model(**tokenized).logits[:, :-1]
                if args.sampling_model_name == args.scoring_model_name:
                    logits_ref = logits_score
                else:
                    tokenized = sampling_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                    assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                    logits_ref = sampling_model(**tokenized).logits[:, :-1]
                original_crit = criterion_fn(logits_ref, logits_score, labels)
            # sampled text
            tokenized = scoring_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
            labels = tokenized.input_ids[:, 1:]
            with torch.no_grad():
                logits_score = scoring_model(**tokenized).logits[:, :-1]
                if args.sampling_model_name == args.scoring_model_name:
                    logits_ref = logits_score
                else:
                    tokenized = sampling_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                    assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                    logits_ref = sampling_model(**tokenized).logits[:, :-1]
                sampled_crit = criterion_fn(logits_ref, logits_score, labels)
        
        # result
        results.append({
            "original": original_text,
            "original_crit": original_crit,
            "sampled": sampled_text,
            "sampled_crit": sampled_crit
        })

    # compute prediction scores for real/sampled passages
    predictions = {
        'real': [x["original_crit"] for x in results],
        'samples': [x["sampled_crit"] for x in results]
    }
    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
    
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    
    # results
    results_file = f'{args.output_file}.{name}.json'
    results = { 
        'name': f'{name}_threshold',
        'info': {'n_samples': n_samples},
        'predictions': predictions,
        'raw_results': results,
        'metrics': {
            'roc_auc': roc_auc, 
            'fpr': fpr, 
            'tpr': tpr
        },
        'pr_metrics': {
            'pr_auc': pr_auc, 
            'precision': p, 
            'recall': r
        },
        'loss': 1 - pr_auc
    }
    with open(results_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {results_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_test/results/xsum_gpt-4.falcon-7b_falcon-7b-instruct")  # output file prefix
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_file', type=str, default="./exp_test/data/xsum_falcon-7b")
    parser.add_argument('--sampling_model_name', type=str, default="falcon-7b")
    parser.add_argument('--scoring_model_name', type=str, default="falcon-7b-instruct")
    parser.add_argument('--discrepancy_analytic', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    experiment(args)