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
import math
from openai import OpenAI
from data_builder import load_data
from model import load_tokenizer, load_model, openai_models
from metrics import get_roc_metrics, get_precision_recall_metrics

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

    for name in criterion_fns:
        criterion_fn = criterion_fns[name]
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        eval_results = []
        
        is_openai_model = args.scoring_model_name in openai_models
        
        for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
            original_text = data["original"][idx]
            sampled_text = data["sampled"][idx]
            
            if is_openai_model:

                original_crit, sampled_crit = 0, 0
                o_tokens, o_logprobs, o_top = scoring_model.score_text_with_logprobs(text=original_text)
                s_tokens, s_logprobs, s_top = scoring_model.score_text_with_logprobs(text=sampled_text)

                if name == 'likelihood':
                    # mean log prob over tokens
                    original_crit = float(np.mean(o_logprobs)) if len(o_logprobs) else float("nan")
                    sampled_crit  = float(np.mean(s_logprobs)) if len(s_logprobs) else float("nan")

                elif name in ('rank', 'logrank'):
                    # approximate rank using top_k alt probabilities
                    def approx_rank(tokens, top_lists):
                        ranks = []
                        for tok, top in zip(tokens, top_lists):
                            found = next((t['rank'] for t in top if t['token'] == tok), None)
                            r = (found + 1) if found is not None else (top_k + 1)
                            ranks.append(float(r))
                        if not ranks:
                            return float("nan")
                        if name == 'rank':
                            return -float(np.mean(ranks))
                        else:
                            return -float(np.mean(np.log(np.array(ranks, dtype=np.float32))))
                    original_crit = approx_rank(o_tokens, o_top)
                    sampled_crit  = approx_rank(s_tokens, s_top)

                elif name == 'entropy':
                    # approximate token-level entropy using the observed top_k distribution (normalized)
                    def approx_entropy(top_lists):
                        ents = []
                        for top in top_lists:
                            if not top:
                                continue
                            lp = np.array([t['logprob'] for t in top], dtype=np.float64)
                            p = np.exp(lp - lp.max())
                            p = p / p.sum() if p.sum() > 0 else p
                            ent = -float(np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))
                            ents.append(ent)
                        return float(np.mean(ents)) if ents else float("nan")
                    original_crit = approx_entropy(o_top)
                    sampled_crit  = approx_entropy(s_top)

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

        # compute prediction scores for real/sampled passages
        predictions = {
            'real': [x["original_crit"] for x in eval_results],
            'samples': [x["sampled_crit"] for x in eval_results]
        }

        fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
        p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
        print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

        # log results
        results_file = f'{args.output_file}.{name}.json'
        results = {
            'name': f'{name}_threshold',
            'info': {
                'n_samples': n_samples
            },
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
