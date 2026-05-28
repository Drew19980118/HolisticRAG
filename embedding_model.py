# embedding_model.py
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import warnings


class BaseEmbeddingModel:
    """Base class for all embedding models."""
    def __init__(self, model_name: str, device: str = "cuda:0"):
        self.model_name = model_name
        self.device = device
        self.embedding_dim = None

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        raise NotImplementedError

# ----------------------------------------------------------------------
# 1. NV-Embed-v2 (models with .encode() method, e.g., nvidia/NV-Embed-v2)
# ----------------------------------------------------------------------
class NVEmbedModel(BaseEmbeddingModel):
    def __init__(self, model_name: str, device: str = "cuda:0", **kwargs):
        super().__init__(model_name, device)
        config_dict = {
            "embedding_model_name": model_name,
            "norm": True,
            "model_init_params": {
                "pretrained_model_name_or_path": model_name,
                "trust_remote_code": True,
                "device_map": device,
                "torch_dtype": "auto",
            },
            "encode_params": {
                "max_length": 2048,
                "instruction": "",
                "batch_size": 8,
                "num_workers": 32,
            },
        }
        # Override with kwargs
        if "model_init_params" in kwargs:
            config_dict["model_init_params"].update(kwargs["model_init_params"])
        if "encode_params" in kwargs:
            config_dict["encode_params"].update(kwargs["encode_params"])
        if "norm" in kwargs:
            config_dict["norm"] = kwargs["norm"]

        self.norm = config_dict["norm"]
        self.encode_params = config_dict["encode_params"]
        print(f"Loading NV-Embed model: {model_name}")
        self.model = AutoModel.from_pretrained(**config_dict["model_init_params"])
        self.embedding_dim = self.model.config.hidden_size
        print(f"Model dimension: {self.embedding_dim}")

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        params = deepcopy(self.encode_params)
        if kwargs:
            params.update(kwargs)

        if "instruction" in params and params["instruction"]:
            params["instruction"] = f"Instruct: {params['instruction']}\nQuery: "

        batch_size = params.pop("batch_size", 8)

        if len(texts) <= batch_size:
            params["prompts"] = texts
            results = self.model.encode(**params)
        else:
            pbar = tqdm(total=len(texts), desc="Batch Encoding")
            results = []
            for i in range(0, len(texts), batch_size):
                batch_params = params.copy()
                batch_params["prompts"] = texts[i:i + batch_size]
                results.append(self.model.encode(**batch_params))
                pbar.update(batch_size)
            pbar.close()
            results = torch.cat(results, dim=0)

        if isinstance(results, torch.Tensor):
            results = results.cpu().numpy()
        if self.norm:
            norms = np.linalg.norm(results, axis=1, keepdims=True)
            results = results / norms
        return results

# ----------------------------------------------------------------------
# 2. E5-mistral (intfloat/e5-mistral-7b-instruct)
# ----------------------------------------------------------------------
class E5MistralModel(BaseEmbeddingModel):
    def __init__(self, model_name: str, device: str = "cuda:0", **kwargs):
        super().__init__(model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float16,
        )
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size
        print(f"Loaded E5-Mistral model: {model_name}, dimension: {self.embedding_dim}")

        self.max_length = kwargs.get("max_length", 2048)
        self.batch_size = kwargs.get("batch_size", 8)

    def _mean_pooling(self, last_hidden_states, attention_mask):
        """Mean pooling ignoring padded tokens."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        batch_size = kwargs.get("batch_size", self.batch_size)
        max_length = kwargs.get("max_length", self.max_length)

        # E5-mistral uses instruction: "Instruct: {task}\nQuery: " but default is just the query
        # We'll add the instruction if provided
        instruction = kwargs.get("instruction", "")
        if instruction:
            texts = [f"Instruct: {instruction}\nQuery: {t}" for t in texts]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Use last hidden state (or second last; E5 often uses last)
                last_hidden = outputs.hidden_states[-1]
                embeddings = self._mean_pooling(last_hidden, inputs["attention_mask"])
                embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalize
            all_embeddings.append(embeddings.cpu())
        result = torch.cat(all_embeddings, dim=0).numpy()
        return result


# ----------------------------------------------------------------------
# Factory function to automatically pick the correct model
# ----------------------------------------------------------------------
def get_embedding_model(model_name: str, device: str = "cuda:0", **kwargs) -> BaseEmbeddingModel:
    """
    Create an embedding model instance based on the model name.
    Supports:
        - NV-Embed-v2 (contains 'NV-Embed' or uses .encode)
        - LLM2Vec (contains 'LLM2Vec')
        - E5-mistral (contains 'e5-mistral')
    """
    name_lower = model_name.lower()
    if "nv-embed" in name_lower:
        return NVEmbedModel(model_name, device=device, **kwargs)
    elif "e5-mistral" in name_lower:
        return E5MistralModel(model_name, device=device, **kwargs)
    else:
        # Default to NVEmbed-like if it has an encode method, else raise error
        warnings.warn(f"Unknown model type '{model_name}'. Attempting to load as NVEmbedModel.")
        return NVEmbedModel(model_name, device=device, **kwargs)