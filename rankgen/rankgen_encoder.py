from transformers import T5Tokenizer, AutoModel
from t5 import T5EncoderModel, T5EncoderWithProjection
import pickle
import argparse
import numpy as np
import tqdm
import os
import torch


class RankGenEncoder(torch.nn.Module):
    def __init__(self, model_path, max_batch_size=32, model_size=None, cache_dir=None):
        super().__init__()
        assert model_path in ["kalpeshk2011/rankgen-t5-xl-all", "kalpeshk2011/rankgen-t5-xl-pg19",
                              "kalpeshk2011/rankgen-t5-base-all", "kalpeshk2011/rankgen-t5-large-all"]
        self.max_batch_size = max_batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if model_size is None:
            if "t5-large" in model_path or "t5_large" in model_path:
                self.model_size = "large"
            elif "t5-xl" in model_path or "t5_xl" in model_path:
                self.model_size = "xl"
            else:
                self.model_size = "base"
        else:
            self.model_size = model_size

        self.tokenizer = T5Tokenizer.from_pretrained(f"google/t5-v1_1-{self.model_size}", cache_dir=cache_dir)
        t5_encoder = T5EncoderModel.from_pretrained(model_path)
        self.model = T5EncoderWithProjection(t5_encoder.config, t5_encoder)
        self.model.to(self.device)

    def encode(self, inputs, learned_vector=None, vectors_type="prefix", return_cat=True, verbose=False, return_input_ids=False):
        tokenizer = self.tokenizer
        max_batch_size = 1
        if isinstance(inputs, str):
            inputs = [inputs]
        if vectors_type == 'prefix':
            inputs = ['pre ' + input for input in inputs]
            max_length = 512
        else:
            inputs = ['suffi ' + input for input in inputs]
            max_length = 128

        all_embeddings = []
        all_input_ids = []
        for i in tqdm.tqdm(range(0, len(inputs), max_batch_size), total=(len(inputs) // max_batch_size) + 1,
                           disable=not verbose, desc=f"Encoding {vectors_type} inputs:"):
            tokenized_inputs = tokenizer(inputs[i:i + max_batch_size], return_tensors="pt", padding=True)
            for k, v in tokenized_inputs.items():
                tokenized_inputs[k] = v[:, :max_length]
            if learned_vector is not None:
                tokenized_inputs['learned_vector'] = learned_vector
            tokenized_inputs = tokenized_inputs.to(self.device)
            batch_embeddings = self.model(**tokenized_inputs)
            all_embeddings.append(batch_embeddings)
            if return_input_ids:
                all_input_ids.extend(tokenized_inputs.input_ids.cpu().tolist())
        all_embeds = [embed[0] for embed in all_embeddings]
        if return_cat:
            all_embeds = torch.cat(all_embeds, dim=0)
            all_embeds = torch.squeeze(all_embeds)

        return {
            "embeddings": all_embeds,
            "input_ids": all_input_ids
        }

class T5XEmbeddingGeneratorLegacy():
    '''This class is deprecated, use RankGenEncoder.'''

    def __init__(self, max_batch_size=32, model_path='.', model_size=None, cache_dir=None):
        self.max_batch_size = max_batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if model_size is None:
            if "t5_large" in model_path:
                self.model_size = "large"
            elif "t5_xl" in model_path:
                self.model_size = "xl"
            else:
                self.model_size = "base"
        else:
            self.model_size = model_size

        with open(os.path.join('', 'state_dict.pickle'), 'rb') as handle:
            state_dict = pickle.load(handle)

        state_dict_new = {}

        for k, v in state_dict.items():
            if k != "encoder.embed_tokens.weight":
                v = np.transpose(v)
                state_dict_new[k] = torch.Tensor(v)
            else:
                state_dict_new[k] = torch.Tensor(v)
                state_dict_new["shared.weight"] = torch.Tensor(v)

        with open(os.path.join('', 'projection.pickle'), 'rb') as handle:
            self.projection = torch.Tensor(pickle.load(handle))  # (1024, 1024), numpy array

        self.projection = self.projection.to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(f"google/t5-v1_1-{self.model_size}", cache_dir=cache_dir)
        model = T5EncoderModel.from_pretrained(f"google/t5-v1_1-{self.model_size}", cache_dir=cache_dir)
        state_dict_keys = [k for k in state_dict_new.keys()]
        self.model, _, _, _, _ = T5EncoderModel._load_pretrained_model(model, state_dict_new, state_dict_keys, None,
                                                                       f"google/t5-v1_1-{self.model_size}")
        self.model.to(self.device)
        self.model.eval()

    def encode(self, inputs, vectors_type="prefix", verbose=False, return_input_ids=False):
        tokenizer = self.tokenizer
        max_batch_size = self.max_batch_size
        if isinstance(inputs, str):
            inputs = [inputs]
        if vectors_type == 'prefix':
            inputs = ['pre ' + input for input in inputs]
            max_length = 512
        else:
            inputs = ['suffi ' + input for input in inputs]
            max_length = 128

        all_embeddings = []
        all_input_ids = []
        for i in tqdm.tqdm(range(0, len(inputs), max_batch_size), total=(len(inputs) // max_batch_size) + 1,
                           disable=not verbose, desc=f"Encoding {vectors_type} inputs:"):
            tokenized_inputs = tokenizer(inputs[i:i + max_batch_size], return_tensors="pt", padding=True)
            for k, v in tokenized_inputs.items():
                tokenized_inputs[k] = v[:, :max_length]
            tokenized_inputs = tokenized_inputs.to(self.device)
            with torch.no_grad():
                hidden_states = self.model(**tokenized_inputs).last_hidden_state
                hidden_states = hidden_states[:, 0, :]
                batch_embeddings = torch.matmul(hidden_states, self.projection)
            all_embeddings.append(batch_embeddings)
            if return_input_ids:
                all_input_ids.extend(tokenized_inputs.input_ids.cpu().tolist())
        return {
            "embeddings": torch.cat(all_embeddings, dim=0),
            "input_ids": all_input_ids
        }
