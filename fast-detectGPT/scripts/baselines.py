# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import argparse
import json
from data_builder import load_data
from model import load_tokenizer, load_model, openai_models
from metrics import get_roc_metrics, get_precision_recall_metrics
from itertools import zip_longest

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_likelihood.mean().item()

def get_rank(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    # get rank of each label token in the model's likelihood ordering
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]

    # make sure we got exactly one match for each timestep in the sequence
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1 # convert to 1-indexed rank
    return -ranks.mean().item()

def get_logrank(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    # get rank of each label token in the model's likelihood ordering
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]

    # make sure we got exactly one match for each timestep in the sequence
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1  # convert to 1-indexed rank
    ranks = torch.log(ranks)
    return -ranks.mean().item()

def get_entropy(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
    entropy = -entropy.sum(-1)
    return entropy.mean().item()

def _mean_ignore_nones(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")

def _align_api_logs(logs, top_k):
    tokens = logs.get("tokens") or []
    tops = logs.get("top_logprobs_per_token") or []
    token_ll = logs.get("token_logprobs") or []
    n = min(len(tokens), len(tops), len(token_ll))
    tokens = tokens[:n]
    tops = [([] if t is None else t[:top_k]) for t in tops[:n]]
    token_ll = [float(x) if x is not None else np.nan for x in token_ll[:n]]
    return tokens, tops, token_ll

def _approx_rank_from_tops(tokens, tops, use_log=False, default_top_k=20):
    ranks = []
    for tok, top in zip_longest(tokens, tops, fillvalue=[]):
        if tok is None:
            continue
        found_rank = None
        for t in (top or []):
            if t.get("token") == tok:
                found_rank = t.get("rank", None)
                break
        k = len(top) if top else default_top_k
        r = (found_rank + 1) if (found_rank is not None) else (k + 1)
        ranks.append(float(r))
    if not ranks:
        return float("nan")
    arr = np.array(ranks, dtype=np.float64)
    return -float(np.mean(np.log(arr))) if use_log else -float(np.mean(arr))

def _approx_entropy_from_tops(tops):
    ents = []
    for top in tops:
        if not top:
            continue
        lp = np.array([t["logprob"] for t in top], dtype=np.float64)
        p = np.exp(lp - lp.max())
        s = p.sum()
        if s <= 0:
            continue
        p = p / s
        ent = -float(np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))
        ents.append(ent)
    return float(np.mean(ents)) if ents else float("nan")

def _filter_finite(xs):
    return [x for x in xs if x is not None and np.isfinite(x)]

def experiment(args):
    
    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval() if hasattr(scoring_model, "eval") else None
    
    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])

    # eval criterions
    criterion_fns = {
        'likelihood': get_likelihood,
        'rank': get_rank,
        'logrank': get_logrank,
        'entropy': get_entropy
    }

    is_openai_model = args.scoring_model_name in openai_models
    top_k, max_length = 20, 300

    for name, criterion_fn in criterion_fns.items():
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        eval_results = []

        for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
            original_text = data["original"][idx]
            sampled_text = data["sampled"][idx]

            if is_openai_model:
                o_logs = scoring_model.score_text_with_logprobs(text=original_text, top_k=top_k, max_length=max_length)
                s_logs = scoring_model.score_text_with_logprobs(text=sampled_text,  top_k=top_k, max_length=max_length)

                o_tokens, o_tops, o_ll = _align_api_logs(o_logs, top_k)
                s_tokens, s_tops, s_ll = _align_api_logs(s_logs, top_k)

                if name == 'likelihood':
                    original_crit = _mean_ignore_nones(o_ll)
                    sampled_crit = _mean_ignore_nones(s_ll)

                elif name in ('rank', 'logrank'):
                    use_log = (name == 'logrank')
                    original_crit = _approx_rank_from_tops(o_tokens, o_tops, use_log=use_log, default_top_k=top_k)
                    sampled_crit = _approx_rank_from_tops(s_tokens, s_tops, use_log=use_log, default_top_k=top_k)

                elif name == 'entropy':
                    original_crit = _approx_entropy_from_tops(o_tops)
                    sampled_crit = _approx_entropy_from_tops(s_tops)

            else:
                # original text
                tokenized = scoring_tokenizer(
                    original_text, 
                    return_tensors="pt", 
                    padding=True, 
                    return_token_type_ids=False).to(args.device)
                labels = tokenized.input_ids[:, 1:]
                with torch.no_grad():
                    logits = scoring_model(**tokenized).logits[:, :-1]
                    original_crit = criterion_fn(logits, labels)
                # sampled text
                tokenized = scoring_tokenizer(
                    sampled_text, 
                    return_tensors="pt", 
                    padding=True, 
                    return_token_type_ids=False).to(args.device)
                labels = tokenized.input_ids[:, 1:]
                with torch.no_grad():
                    logits = scoring_model(**tokenized).logits[:, :-1]
                    sampled_crit = criterion_fn(logits, labels)
            
            # result
            eval_results.append({
                "original": original_text,
                "original_crit": original_crit,
                "sampled": sampled_text,
                "sampled_crit": sampled_crit
            })

        predictions = {
            'real': _filter_finite([x["original_crit"] for x in eval_results]),
            'samples': _filter_finite([x["sampled_crit"]  for x in eval_results])
        }

        fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
        p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
        print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

        results_file = f'{args.output_file}.{name}.json'
        results = {
            'name': f'{name}_threshold',
            'info': {'n_samples': n_samples},
            'predictions': predictions,
            'raw_results': eval_results,
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
    parser.add_argument('--output_file', type=str, default="./exp_test/results/xsum_gpt2")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_file', type=str, default="./exp_test/data/xsum_gpt2")
    parser.add_argument('--scoring_model_name', type=str, default="gpt2")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()
    experiment(args)
