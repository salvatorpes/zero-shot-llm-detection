import torch
import numpy as np
from openai import OpenAI
import tiktoken
import time
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).with_name('.env'))
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

GPT_MODEL_DATA = {
    'gpt-4o': {
        'context_tokens': 128000, 
        'costs': {
            'input_cost': 2.5 / 1000000, 
            'output_cost': 10 / 1000000
        },
        'model_options': {
            "n": 1,
            "stop": None,
            "max_completion_tokens": 16384,
            "logprobs": True,
            "top_logprobs": 0
        }
    },
    'gpt-4o-mini': {
        'context_tokens': 128000, 
        'costs': {
            'input_cost': 0.15 / 1000000, 
            'output_cost': 0.6 / 1000000
        },
        'model_options': {
            "n": 1,
            "stop": None,
            "max_completion_tokens": 16384,
            "logprobs": True,
            "top_logprobs": 0
        }
    },
    'gpt-5': {
        'context_tokens': 400000, 
        'costs': {
            'input_cost': 1.25 / 1000000, 
            'output_cost': 10 / 1000000
        },
        'model_options': {
            "n": 1,
            "stop": None,
            "max_completion_tokens": 128000,
            "logprobs": True,
            "top_logprobs": 0
        }
    },
    'gpt-5-mini': {
        'context_tokens': 400000, 
        'costs': {
            'input_cost': 0.25 / 1000000, 
            'output_cost': 2 / 1000000
        },
        'model_options': {
            "n": 1,
            "stop": None,
            "max_completion_tokens": 128000,
            "logprobs": True,
            "top_logprobs": 0
        }
    },
    'gpt-5-nano': {
        'context_tokens': 400000, 
        'costs': {
            'input_cost': 0.05 / 1000000, 
            'output_cost': 0.4 / 1000000
        },
        'model_options': {
            "n": 1,
            "stop": None,
            "max_completion_tokens": 128000,
            "logprobs": True,
            "top_logprobs": 0
        }
    },
}

class OpenAIModel:
    
    def __init__(self, model):
        self.model = model
        if self.model not in ['gpt-4o', 'gpt-4o-mini', 'gpt-5', 'gpt-5-mini', 'gpt-5-nano']:
            raise ValueError(f"Model {self.model} is not supported")
        
        self.api_key = OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("Must provide OpenAI API key via .env file") 

        self.base_url = OPENAI_BASE_URL
        self.org_id = OPENAI_ORG_ID
        
        client_kwargs = {'api_key': self.api_key}
        if self.base_url:
            client_kwargs['base_url'] = self.base_url
        if self.org_id:
            client_kwargs['organization'] = self.org_id
            
        self.client = OpenAI(**client_kwargs)

        try:
            if self.model.startswith('gpt-4o'):
                self.tokenizer = tiktoken.get_encoding("gpt-4o")
            elif self.model.startswith('gpt-5'):
                self.tokenizer = tiktoken.get_encoding("gpt-5")
            else:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.models_data = GPT_MODEL_DATA
        self.safeguard_model = 'gpt-4o-mini'
    
    def sample_from_openai(
        self, 
        prefix, 
        dataset, 
        do_top_p=False, 
        top_p=1.0, 
        do_top_k=False, 
        top_k=0.0, 
        do_temperature=False, 
        temperature=1.0
    ):

        def _drop_last_word(text):
            parts = text.split(' ')
            return ' '.join(parts[:-1]) if len(parts) > 1 else text

        if dataset != 'pubmed':
            prefix = _drop_last_word(prefix)

        total_input_tokens, total_output_tokens = 0, 0

        options = self.models_data.get(self.model, self.models_data.get(self.safeguard_model, {})).get('model_options', {})
        options["max_completion_tokens"] = 300
        if do_temperature:
            options["temperature"] = temperature
        if do_top_p and top_p < 1.0:
            options["top_p"] = top_p
        if do_top_k and top_k > 0.0:
            if top_k > 20.0:
                raise ValueError("top_k must be less than or equal to 20")
            options["top_logprobs"] = int(top_k)

        roles = {
            'xsum': 'You are a News writer.',
            'writing': 'You are a Fiction writer.',
            'pubmed': 'You are a Technical writer.',
            'hc3': 'You are a Technical writer.'
        }
        prompts = {
            'xsum': 'Please write an article with about 150 words starting exactly with:',
            'writing': 'Please write an article with about 150 words starting exactly with:',
            'pubmed': 'Please answer the question in about 50 words.',
            'hc3': 'Please write a continuation of the following text in about 150 words starting exactly with:'
        }
        messages = [
            {'role': 'system', 'content': roles[dataset]},
            {'role': 'user', 'content': f'{prompts[dataset]} {prefix}'},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **options
            )
            generated_text = response.choices[0].message.content.strip()
            total_input_tokens += response.usage.prompt_tokens
            total_output_tokens += response.usage.completion_tokens
            logprobs = self.process_logprobs(response.choices[0].logprobs, separate=True)
        except Exception as e:
            raise Exception(f"API error occurred: {e}")

        return generated_text, logprobs

    def score_text_with_logprobs(
        self,
        text,
        top_k=20,
        max_length=300
    ):

        system_prompt = (
            "Repeat EXACTLY the following text, character by character, with no changes. "
            "Do not add quotes or commentary. Output only the text."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]

        total_input_tokens, total_output_tokens = 0, 0

        options = self.models_data.get(self.model, self.models_data.get(self.safeguard_model, {})).get('model_options', {})
        options.update({
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": int(top_k),
            "max_completion_tokens": int(max_length),
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **options
            )
            generated_text = response.choices[0].message.content.strip()
            total_input_tokens += response.usage.prompt_tokens
            total_output_tokens += response.usage.completion_tokens
            logprobs = self.process_logprobs(response.choices[0].logprobs, separate=True)
        except Exception as e:
            raise Exception(f"API error during scoring: {e}")

        return logprobs

    def cost_calculator(self, input_tokens=0, output_tokens=0):
        try:
            model_costs = self.models_data.get(self.model, {}).get('costs', {})
            return input_tokens * model_costs['input_cost'] + output_tokens * model_costs['output_cost']
        except Exception as e:
            raise Exception(f"Failed to calculate the messages cost: {e}.")
    
    def process_logprobs(self, logprobs, separate=False):
        logprobs_list = []
        if not logprobs or not getattr(logprobs, "content", None):
            return [] if not separate else {"tokens": [], "token_logprobs": [], "top_logprobs_per_token": []}

        for logprob in logprobs.content:
            logprobs_list.append({
                "token": logprob.token,
                "logprob": logprob.logprob,
                "top": [{"token": n.token, "logprob": n.logprob, "rank": i} for i, n in enumerate(logprob.top_logprobs)]	
            })

        if not separate:
            return logprobs_list
        return {
            "tokens": [i["token"] for i in logprobs_list],
            "token_logprobs": [i["logprob"] for i in logprobs_list],
            "top_logprobs_per_token": [i["top"] for i in logprobs_list]
        }        
    
    def eval(self):
        return self
    
    def encode(self, text):
        return self.tokenizer.encode(text)

    def encode_batch(self, texts):
        return self.tokenizer.encode_batch(texts)

    def decode(self, ids):
        return self.tokenizer.decode(ids)