import torch
import numpy as np
from openai import OpenAI
import tiktoken
import time
import os
from dotenv import load_dotenv

load_dotenv()
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
        
        if 'gpt-4' in model:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        elif 'gpt-3.5' in model:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.models_data = GPT_MODEL_DATA
        self.safeguard_model = 'gpt-4o-mini'
    
    def generate_message(
        self, 
        input_ids=None, 
        texts=None,
        max_length=250, 
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0.0
    ):

        if texts is None and input_ids is not None:
            if torch.is_tensor(input_ids):
                input_ids = input_ids.cpu().numpy()
            texts = [self.tokenizer.decode(ids) for ids in input_ids]
        
        if texts is None:
            raise ValueError("Either 'texts' or 'input_ids' must be provided")
        
        generated_texts = []
        total_input_tokens, total_output_tokens = 0, 0
        logprobs = None
        
        for text in texts:

            messages = [{
                    "role": "user", 
                    "content": f"Continue the following text: {text}"
                }]
            options = self.models_data.get(self.model, self.models_data.get(self.safeguard_model, {})).get('model_options', {})
            options["temperature"] = temperature
            if do_sample and top_p < 1.0:
                options["top_p"] = top_p
            if do_sample and top_k > 0.0:
                if top_k > 20:
                    raise ValueError("top_k must be less than or equal to 20")
                options["top_logprobs"] = int(top_k)

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=max_length,
                    **options
                )
                generated_text = response.choices[0].message.content.strip()
                logprobs = self.process_logprobs(response.choices[0].logprobs)
                total_input_tokens += response.usage.prompt_tokens
                total_output_tokens += response.usage.completion_tokens
            except Exception as e:
                raise Exception(f"API error occurred: {e}")
             
            generated_texts.append(generated_text)
        
        return generated_text, logprobs, total_input_tokens, total_output_tokens
    
    def sample_from_model(
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
            return ' '.join(text.split(' ')[:-1])

        if dataset != 'pubmed':
            prefix = _drop_last_word(prefix)

        options = self.models_data.get(self.model, self.models_data.get(self.safeguard_model, {})).get('model_options', {})
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
            'pubmed': 'You are a Technical writer.'
        }
        prompts = {
            'xsum': 'Please write an article with about 150 words starting exactly with:',
            'writing': 'Please write an article with about 150 words starting exactly with:',
            'pubmed': 'Please answer the question in about 50 words.'
        }
        messages = [
            {'role': 'system', 'content': roles[dataset]},
            {'role': 'user', 'content': f'{prompts[dataset]} {prefix}'},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=300,
                **options
            )
            generated_text = response.choices[0].message.content.strip()
            logprobs = self.process_logprobs(response.choices[0].logprobs)
        except Exception as e:
            raise Exception(f"API error occurred: {e}")

        return generated_text, logprobs

    def score_text_with_logprobs(
        self,
        text,
        top_k=20,
        max_length=300
    ):

        target_ids = self.encode(text)
        max_completion_tokens = min(len(target_ids), max_length)

        system_prompt = (
            "Repeat EXACTLY the following text, character by character, with no changes. "
            "Do not add quotes or commentary. Output only the text."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]

        options = self.models_data.get(self.model, self.models_data.get(self.safeguard_model, {})).get('model_options', {})
        options.update({
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": int(top_k),
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                **options
            )
            generated_text = response.choices[0].message.content.strip()
            tokens, token_logprobs, top_logprobs_per_token = self.process_logprobs(response.choices[0].logprobs, return_separate_lists=True)
        except Exception as e:
            raise Exception(f"API error during scoring: {e}")

        return tokens, token_logprobs, top_logprobs_per_token

    def cost_calculator(self, input_tokens=0, output_tokens=0):
        try:
            model_costs = self.models_data.get(self.model, {}).get('costs', {})
            return input_tokens * model_costs['input_cost'] + output_tokens * model_costs['output_cost']
        except Exception as e:
            raise Exception(f"Failed to calculate the messages cost: {e}.")
    
    def process_logprobs(self, logprobs, return_separate_lists=False):
        logprobs_list = []
        if not logprobs:
            return logprobs_list
        for rank,logprob in enumerate(logprobs.content):
            logprobs_list.append({
                "token": logprob.token,
                "logprob": logprob.logprob
                "rank": rank	
            })
        if return_separate_lists:
            tokens = [i["token"] for i in logprobs_list]
            token_logprobs = [i["logprob"] for i in logprobs_list]
            top_logprobs_per_token = [i["rank"] for i in logprobs_list]
            return tokens, token_logprobs, top_logprobs_per_token
        else:
            return logprobs_list
    
    def eval(self):
        return self
    
    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, ids, skip_special_tokens=True):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    
    def batch_decode(self, sequences, skip_special_tokens=True, **kwargs):
        if torch.is_tensor(sequences):
            sequences = sequences.cpu().numpy().tolist()
        
        results = []
        for seq in sequences:
            if skip_special_tokens:
                seq = [token for token in seq if token != self.pad_token_id]
            results.append(self.decode(seq, skip_special_tokens=skip_special_tokens))
        
        return results
