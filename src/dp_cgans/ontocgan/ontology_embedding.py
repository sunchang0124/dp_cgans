
from gensim.models import KeyedVectors
from typing import Optional
import numpy as np
import os, json

# ---------------------------
# Existing OntologyEmbedding
# ---------------------------
class OntologyEmbedding:
    """Ontology embedding class using Gensim KeyedVectors (with JSON cache)."""
    def __init__(self, embedding_path, embedding_size, hp_dict_fn, rd_dict_fn,
                 cache_path: Optional[str] = None):
        self._embed_model = KeyedVectors.load(embedding_path)
        self.embed_size = getattr(self._embed_model, "vector_size", embedding_size)
        self.cache = _LocalEmbeddingCache(cache_path) if cache_path else None  # NEW

        self._iri_dict = {}
        with open(hp_dict_fn) as f:
            for line in f:
                (entity, iri) = line.strip().split(';')
                self._iri_dict[entity] = iri

        with open(rd_dict_fn) as f:
            for line in f:
                (entity, iri) = line.strip().split(';')
                self._iri_dict[entity] = iri

    def get_embedding(self, disease):
        key = disease #" ".join(str(entity).split())                      # NEW (stable key)
        # print(self._embed_model.wv["http://www.orpha.net/ORDO/Orphanet_513"])
        # 1) cache hit
        if self.cache:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        # 2) fetch from KeyedVectors
        try:
            vec = self._embed_model.wv[key]
            
        except KeyError:
            vec = np.zeros(self.embed_size, dtype=np.float32)

        # 3) persist
        if self.cache:
            self.cache.put(key, vec)
            self.cache.save()

        return vec

# class OntologyEmbedding:
#     """Ontology embedding class using Gensim KeyedVectors."""
#     def __init__(self, embedding_path, embedding_size, hp_dict_fn, rd_dict_fn):
#         self._embed_model = KeyedVectors.load(embedding_path)
#         self.embed_size = getattr(self._embed_model, "vector_size", embedding_size)
#         self._iri_dict = {}

#         with open(hp_dict_fn) as f:
#             for line in f:
#                 (entity, iri) = line.strip().split(';')
#                 self._iri_dict[entity] = iri

#         with open(rd_dict_fn) as f:
#             for line in f:
#                 (entity, iri) = line.strip().split(';')
#                 self._iri_dict[entity] = iri

#     def get_embedding(self, entity):
#         # If you later want to try IRI fallbacks, add them here.
#         try:
#             return self._embed_model.wv[entity]
#         except KeyError:
#             return np.zeros(self.embed_size, dtype=np.float32)


# ---------------------------
# Local JSON cache helper
# ---------------------------
class _LocalEmbeddingCache:
    def __init__(self, path: str ):
        self.path = path
        self._m = {}
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._m = {k: np.array(v, dtype=np.float32) for k, v in raw.items()}

    def get(self, k):
        return self._m.get(k) if self._m is not None else None

    def put(self, k, v: np.ndarray):
        if self._m is not None:
            self._m[k] = v.astype(np.float32)

    def save(self):
        if self.path and self._m is not None:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({k: v.tolist() for k, v in self._m.items()}, f)


# ---------------------------
# LLMEmbedding (OpenAI or SentenceTransformers)
# ---------------------------
class LLMEmbedding:
    """
    LLM-based embedding class with optional local persistence.
    - backend="openai": uses OpenAI embeddings (API)
    - backend="phi" or "sentence-transformers": local model (no API)
    """
    def __init__(
        self,
        backend: str = "openai",
        cache_path: str = "llm_embeddings.json",
        normalize: bool = False,
        **backend_kwargs
    ):
        self.backend = backend
        self.normalize = normalize
        self.cache = _LocalEmbeddingCache(cache_path)

        if backend == "openai":
            # backend_kwargs: model="text-embedding-3-small" | "text-embedding-3-large", etc.
            from openai import OpenAI
            self._client = OpenAI()
            self._model = backend_kwargs.get("model", "text-embedding-3-small")
            self.embed_size = 3072 if "large" in self._model else 1536
            self._encode_fn = self._encode_openai

        # elif backend in ("sentence-transformers", "st"):
        #     # backend_kwargs: model_name="sentence-transformers/all-MiniLM-L6-v2", ...
        #     from sentence_transformers import SentenceTransformer
        #     model_name = backend_kwargs.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        #     self._st_model = SentenceTransformer(model_name)
        #     self.embed_size = int(self._st_model.get_sentence_embedding_dimension())
        #     self._encode_fn = self._encode_st

        elif backend in ("hf-local", "huggingface-local"):
            # backend_kwargs:
            #   model_name_or_path (str)  -> HF repo id or local dir (required)
            #   device (str)              -> "cuda" | "cpu" (default: auto)
            #   dtype (str)               -> "float16" | "bfloat16" | "float32" (default: float32)
            #   pooling (str)             -> "mean" | "cls" (default: "mean")
            #   max_length (int)          -> tokenizer max length (default: 512)
            #   instruction (str)         -> optional prefix (e.g., "passage: " for E5 models)
            from transformers import AutoModel, AutoTokenizer
            import torch

            model_name = backend_kwargs.get("model_name_or_path")
            if not model_name:
                raise ValueError("hf-local backend requires backend_kwargs['model_name_or_path'].")

            device = backend_kwargs.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

            dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
            dtype = dtype_map.get(str(backend_kwargs.get("dtype", "float32")).lower(), torch.float32)

            self._hf_pooling = str(backend_kwargs.get("pooling", "mean")).lower()
            self._hf_max_len = int(backend_kwargs.get("max_length", 512))
            self._hf_instruction = backend_kwargs.get("instruction", "")

            self._hf_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self._hf_model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
            self._hf_device = device

            hidden_size = getattr(self._hf_model.config, "hidden_size", None)
            if hidden_size is None:
                with torch.inference_mode():
                    toks = self._hf_tokenizer("probe", return_tensors="pt", truncation=True, max_length=16)
                    toks = {k: v.to(self._hf_device) for k, v in toks.items()}
                    out = self._hf_model(**toks)
                    hidden_size = out.last_hidden_state.shape[-1]
            self.embed_size = int(hidden_size)

            self._encode_fn = self._encode_hf_local


        else:
            raise ValueError("backend must be 'openai' or 'other huggingface models'")

    def _encode_hf_local(self, text: str) -> np.ndarray:
        import torch, numpy as np
        prompt = f"{self._hf_instruction}{text}" if self._hf_instruction else text
        toks = self._hf_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self._hf_max_len)
        toks = {k: v.to(self._hf_device) for k, v in toks.items()}
        with torch.inference_mode():
            out = self._hf_model(**toks)
            hidden = out.last_hidden_state  # [1, T, H]
            if self._hf_pooling == "cls":
                vec = hidden[:, 0, :]
            else:
                attn = toks.get("attention_mask", torch.ones(hidden.shape[:2], device=hidden.device)).unsqueeze(-1)
                vec = (hidden * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1)
        return vec[0].detach().cpu().to(torch.float32).numpy()

    
    
    def _normalize(self, v: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return v
        n = np.linalg.norm(v)
        return v / (n + 1e-12)

    def _encode_openai(self, text: str) -> np.ndarray:
        text = " ".join(str(text).split())
        resp = self._client.embeddings.create(model=self._model, input=text)
        return np.array(resp.data[0].embedding, dtype=np.float32)

    def _encode_st(self, text: str) -> np.ndarray:
        return self._st_model.encode([str(text)], convert_to_numpy=True)[0].astype(np.float32)

    def get_embedding(self, entity) -> np.ndarray:
        if isinstance(entity, str):
            key = " ".join(str(entity).split())

        elif isinstance(entity, tuple) and len(entity) == 2:
            # (key, value) from a dict iteration
            k, v = entity
            # you decide which one to embed — here I’ll embed the value
            key = " ".join(str(v).split())
    
        else:
            raise ValueError("Entity must be a str or a (key,value) tuple.")

        
        # 1) cache
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        # 2) compute
        try:
            vec = self._encode_fn(v)
        except Exception:
            # Fail-safe: consistent shape zeros
            return np.zeros(self.embed_size, dtype=np.float32)

        vec = self._normalize(vec)

        # 3) persist
        self.cache.put(key, vec)
        self.cache.save()
        return vec


# ---------------------------
# Simple factory helper
# ---------------------------
def make_embedding_model(
    use_llm: bool,
    *,
    # OntologyEmbedding (gensim) args
    embedding_path: Optional[str] = None,
    embedding_size: Optional[int] = None,
    hp_dict_fn: Optional[str] = None,
    rd_dict_fn: Optional[str] = None,
    # LLMEmbedding args
    llm_backend: str = "openai",
    cache_path: Optional[str] = None,   # now used by BOTH backends
    normalize: bool = False,
    **llm_kwargs
):
    """
    Returns either OntologyEmbedding or LLMEmbedding, same get_embedding() API.
    """
    if use_llm:
        return LLMEmbedding(
            backend=llm_backend,
            cache_path=cache_path,
            normalize=normalize,
            **llm_kwargs
        )
    else:
        if not all([embedding_path, embedding_size, hp_dict_fn, rd_dict_fn]):
            raise ValueError("Provide embedding_path, embedding_size, hp_dict_fn, rd_dict_fn for OntologyEmbedding.")
        return OntologyEmbedding(
            embedding_path, embedding_size, hp_dict_fn, rd_dict_fn,
            cache_path=cache_path
        )




# from gensim.models import KeyedVectors
# import numpy as np


# class OntologyEmbedding():
#     """Ontology embedding class.

#     This class is specifically designed for embeddings with the RDs + HPs datasets.

#     Args:
#     embedding_path (str):
#         Path to the embeddings file, to be loaded with KeyedVectors (.embedding file).
#     embedding_size (int):
#         Size of each embedding vector.
#     hp_dict_fn (str):
#         Path to the HP label -> HP URI dictionary file.
#     rd_dict_fn (str):
#         Path to the RD label -> RD URI dictionary file.
#     """

#     def __init__(self, embedding_path, embedding_size, hp_dict_fn, rd_dict_fn):
#         self._embed_model = KeyedVectors.load(embedding_path)
#         self.embed_size = embedding_size
#         self._iri_dict = {}

#         with open(hp_dict_fn) as f:
#             for line in f:
#                 (entity, iri) = line.strip().split(';')
#                 self._iri_dict[entity] = iri

#         with open(rd_dict_fn) as f:
#             for line in f:
#                 (entity, iri) = line.strip().split(';')
#                 self._iri_dict[entity] = iri



#     def get_embedding(self, entity):
#         """Get embedding of entity.

#         Args:
#             entity (str):
#                 Label or URI of the class to get the embedding of.
#         Returns:
#             (numpy.ndarray): Embedding of the entity, array full of zeros if embedding not found.
#         """
#         # iri = self.get_iri(entity)
#         # if iri is not None:
#         #     return self._embed_model.wv[iri]

#         # print("http://www.orpha.net/ORDO/Orphanet_"+str(int(entity)))

#         # try: 
#         return self._embed_model.wv[entity]#["http://www.orpha.net/ORDO/Orphanet_"+str(int(entity))]
#         # except:
#         #     print("The disease iri %s is not found in the embeddings." %entity)
#         #     return np.zeros(self.embed_size)
