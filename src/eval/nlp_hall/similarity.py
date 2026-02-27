
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from config.logistics import HallEvalConfig
from src.eval.nlp_hall.normalize import normalize_text


class SimilarityBackend(Protocol):
    def similarity_matrix(self, a: list[str], b: list[str]) -> np.ndarray:
        ...


@dataclass
class SbertSimilarityBackend:
    cfg: HallEvalConfig
    model: object
    _cache: dict[str, np.ndarray] = field(default_factory=dict)

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        # sentence-transformers encode returns L2-normalized vectors when requested.
        vectors = self.model.encode(
            texts,
            batch_size=max(1, int(self.cfg.batch_size)),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.cfg.device,
        )
        return np.asarray(vectors, dtype=np.float32)

    def _vectors(self, texts: list[str]) -> np.ndarray:
        normalized = [normalize_text(x) for x in texts]
        missing = [t for t in normalized if t and t not in self._cache]
        if missing:
            vecs = self._encode_batch(missing)
            for text, vec in zip(missing, vecs, strict=True):
                self._cache[text] = vec.astype(np.float32, copy=False)

        out: list[np.ndarray] = []
        for text in normalized:
            if not text:
                # lazily create a zero vector using any cached vector shape.
                if self._cache:
                    dim = next(iter(self._cache.values())).shape[0]
                    out.append(np.zeros((dim,), dtype=np.float32))
                else:
                    dummy = self._encode_batch([" "])[0]
                    self._cache[" "] = dummy
                    out.append(np.zeros_like(dummy, dtype=np.float32))
            else:
                out.append(self._cache[text])
        return np.vstack(out).astype(np.float32, copy=False)

    def similarity_matrix(self, a: list[str], b: list[str]) -> np.ndarray:
        if not a or not b:
            return np.zeros((len(a), len(b)), dtype=np.float32)
        va = self._vectors(a)
        vb = self._vectors(b)
        sim = va @ vb.T
        np.clip(sim, 0.0, 1.0, out=sim)
        return sim.astype(np.float32, copy=False)


@dataclass
class TfidfSimilarityBackend:
    _pair_cache: dict[tuple[tuple[str, ...], tuple[str, ...]], np.ndarray] = field(default_factory=dict)

    def similarity_matrix(self, a: list[str], b: list[str]) -> np.ndarray:
        if not a or not b:
            return np.zeros((len(a), len(b)), dtype=np.float32)

        a_norm = tuple(normalize_text(x) for x in a)
        b_norm = tuple(normalize_text(x) for x in b)
        cache_key = (a_norm, b_norm)
        cached = self._pair_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except Exception:  # noqa: BLE001
            return _lexical_fallback(a_norm, b_norm)

        corpus = list(a_norm) + list(b_norm)
        vectorizer = TfidfVectorizer(dtype=np.float32)
        mat = vectorizer.fit_transform(corpus)
        a_mat = mat[: len(a_norm)]
        b_mat = mat[len(a_norm) :]
        sim = (a_mat @ b_mat.T).toarray().astype(np.float32, copy=False)
        np.clip(sim, 0.0, 1.0, out=sim)

        self._pair_cache[cache_key] = sim
        return sim.copy()


def _lexical_fallback(a: tuple[str, ...], b: tuple[str, ...]) -> np.ndarray:
    out = np.zeros((len(a), len(b)), dtype=np.float32)
    b_sets = [set(x.split()) for x in b]
    for i, text in enumerate(a):
        a_set = set(text.split())
        for j, b_set in enumerate(b_sets):
            if not a_set and not b_set:
                out[i, j] = 1.0
                continue
            denom = len(a_set | b_set)
            out[i, j] = float(len(a_set & b_set) / denom) if denom else 0.0
    return out


def get_similarity_backend(cfg: HallEvalConfig) -> SimilarityBackend:
    backend = (cfg.backend or "sbert").strip().lower()

    if backend == "sbert":
        try:
            from sentence_transformers import SentenceTransformer

            model_name = cfg.model_name or "sentence-transformers/all-MiniLM-L6-v2"
            model = SentenceTransformer(model_name, device=cfg.device)
            return SbertSimilarityBackend(cfg=cfg, model=model)
        except Exception:  # noqa: BLE001
            return TfidfSimilarityBackend()

    if backend == "tfidf":
        return TfidfSimilarityBackend()

    return TfidfSimilarityBackend()
