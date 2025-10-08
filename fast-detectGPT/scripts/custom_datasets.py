import os.path
import random
import datasets
import tempfile

SEPARATOR = '<<<SEP>>>'


DATASETS = ['writing', 'english', 'german', 'pubmed', 'hc3', 'hc3_all', 'hc3_open_qa', 'hc3_finance', 'hc3_medicine', 'hc3_reddit_eli5', 'hc3_wiki_csai']

def load_dataset(path, name=None, split=None, cache_dir=None):
    # use local model if it exists
    local_path = os.path.join(cache_dir, f'local.{path}_{name}_{split}')
    if os.path.exists(local_path):
        return datasets.load_from_disk(local_path)
    
    # Handle HC3 dataset which doesn't support trust_remote_code parameter
    if 'HC3' in path:
        return datasets.load_dataset(path, name, split=split, cache_dir=cache_dir)
    else:
        return datasets.load_dataset(path, name, split=split, cache_dir=cache_dir, trust_remote_code=True)

def load_pubmed(cache_dir):
    data = load_dataset('pubmed_qa', 'pqa_labeled', split='train', cache_dir=cache_dir)
    
    # combine question and long_answer
    data = [f'Question: {q} Answer:{SEPARATOR}{a}' for q, a in zip(data['question'], data['long_answer'])]

    return data


def process_prompt(prompt):
    return prompt.replace('[ WP ]', '').replace('[ OT ]', '')


def process_spaces(story):
    return story.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()


def load_writing(cache_dir=None):
    writing_path = 'data/writingPrompts'
    
    with open(f'{writing_path}/valid.wp_source', 'r') as f:
        prompts = f.readlines()
    with open(f'{writing_path}/valid.wp_target', 'r') as f:
        stories = f.readlines()
    
    prompts = [process_prompt(prompt) for prompt in prompts]
    joined = [process_spaces(prompt + " " + story) for prompt, story in zip(prompts, stories)]
    filtered = [story for story in joined if 'nsfw' not in story and 'NSFW' not in story]

    random.seed(0)
    random.shuffle(filtered)

    return filtered

def load_hc3(cache_dir=None):
    dataset = datasets.load_dataset("Hello-SimpleAI/HC3", "all", split="train", cache_dir=cache_dir)
    return [entry[0] for entry in dataset["human_answers"]]

def load_language(language, cache_dir):
    # load either the english or german portion of the wmt16 dataset
    assert language in ['en', 'de']
    d = load_dataset('wmt16', 'de-en', split='train', cache_dir=cache_dir)
    docs = d['translation']
    desired_language_docs = [d[language] for d in docs]
    lens = [len(d.split()) for d in desired_language_docs]
    sub = [d for d, l in zip(desired_language_docs, lens) if l > 100 and l < 150]
    return sub


def load_german(cache_dir):
    return load_language('de', cache_dir)


def load_hc3(cache_dir, subsets=None, max_samples_per_subset=1000):
    """Load HC3 dataset - simple version"""
    if subsets is None:
        subsets = ['open_qa']  # Default to just one subset
    
    human_texts = []
    
    for config in subsets:
        try:
            print(f"Loading HC3 config: {config}")
            data = datasets.load_dataset('Hello-SimpleAI/HC3', config, split='train', cache_dir=cache_dir)
            
            count = 0
            for example in data:
                if example.get('human_answers') and example['human_answers']:
                    answer = example['human_answers'][0].strip()
                    human_texts.append(answer)
                    count += 1
                    if count >= max_samples_per_subset:
                        break
            
            print(f"Loaded {count} samples from {config}")
            
        except Exception as e:
            print(f"Failed to load {config}: {e}")
            
    return human_texts


def load_hc3_all(cache_dir):
    """Load from all HC3 subsets"""
    return load_hc3(cache_dir, subsets=['open_qa', 'finance', 'medicine', 'reddit_eli5', 'wiki_csai'], max_samples_per_subset=1000)

def load_hc3_open_qa(cache_dir):
    """Load from HC3 open_qa subset only"""
    return load_hc3(cache_dir, subsets=['open_qa'], max_samples_per_subset=5000)

def load_hc3_finance(cache_dir):
    """Load from HC3 finance subset only"""
    return load_hc3(cache_dir, subsets=['finance'], max_samples_per_subset=5000)

def load_hc3_medicine(cache_dir):
    """Load from HC3 medicine subset only"""
    return load_hc3(cache_dir, subsets=['medicine'], max_samples_per_subset=5000)

def load_hc3_reddit_eli5(cache_dir):
    """Load from HC3 reddit_eli5 subset only"""
    return load_hc3(cache_dir, subsets=['reddit_eli5'], max_samples_per_subset=5000)

def load_hc3_wiki_csai(cache_dir):
    """Load from HC3 wiki_csai subset only"""
    return load_hc3(cache_dir, subsets=['wiki_csai'], max_samples_per_subset=5000)


def load_english(cache_dir):
    return load_language('en', cache_dir)


def load(name, cache_dir, **kwargs):
    if name in DATASETS:
        load_fn = globals()[f'load_{name}']
        return load_fn(cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError(f'Unknown dataset {name}. Available datasets: {DATASETS}')