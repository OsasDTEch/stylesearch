"""Hybrid dense + BM25 search over textured hairstyles, with facet aware boosting.

Architecture (the same shape I use in production retrieval systems):
  1. BM25 over tokenized style documents  -> exact term signal ("4c", "knotless", "no heat")
  2. Dense embeddings (fastembed, BAAI/bge-small-en-v1.5) -> semantic signal
     ("something I will not have to touch for two months" ~ long lasting protective styles)
  3. Facet extraction from the query -> hard domain knowledge boosts
     (hair type compatibility, heat, manipulation, length, swim, occasion)
  4. Weighted score fusion, with a per result explanation of WHY it matched.

If fastembed's model cannot be downloaded (offline machine), the engine falls
back to a TF-IDF + SVD latent semantic vectorizer so the demo still runs.
The active backend is reported in every API response for honesty.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

DATA = Path(__file__).parent / "data" / "styles.json"

# ---------------------------------------------------------------- tokenization

_token_re = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _token_re.findall(text.lower())


def doc_text(s: dict) -> str:
    """Flatten a style record into one searchable document."""
    parts = [
        s["name"], s.get("aka", ""), s["category"], s["description"],
        " ".join(s["hair_types"]),
        " ".join(s.get("occasions", [])),
        "protective style" if s["protective"] else "not protective",
        "no heat heat free" if not s["heat_required"] else "requires heat",
        f"{s['manipulation']} manipulation",
        " ".join(s.get("length_achieved", [])),
        f"lasts {s['lasts_weeks']} weeks",
        "swim friendly" if s.get("swim_friendly") else "",
        "workout gym friendly" if s.get("workout_friendly") else "",
    ]
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------- facet parsing

HAIR_TYPES = ["3a", "3b", "3c", "4a", "4b", "4c"]

LENGTH_WORDS = {
    "shoulder": "shoulder", "shoulders": "shoulder",
    "waist": "waist", "mid back": "mid back", "midback": "mid back",
    "long": "mid back", "short": "short", "bob": "short",
}

OCCASION_WORDS = {
    "work": "work", "office": "work", "interview": "interview",
    "professional": "work", "corporate": "work",
    "wedding": "wedding", "bride": "wedding", "formal": "formal",
    "church": "church", "gym": "gym", "workout": "gym", "sports": "gym",
    "vacation": "vacation", "holiday": "vacation", "travel": "vacation",
    "festival": "festival", "concert": "festival",
    "date": "date night", "birthday": "birthday",
    "photoshoot": "photoshoot", "graduation": "formal",
}


def extract_facets(query: str) -> dict:
    q = query.lower()
    f: dict = {}

    types = [t for t in HAIR_TYPES if re.search(rf"\b{t}\b", q)]
    if types:
        f["hair_types"] = types

    if re.search(r"\bno heat\b|\bheat free\b|without heat|heatless", q):
        f["no_heat"] = True
    if re.search(r"low manipulation|low maintenance|don'?t (want|have) to (touch|do)|lazy|minimal effort|low effort"
                 r"|won'?t have to (think|worry|touch)|set (it )?and forget|hands? off|leave (it )?alone", q):
        f["low_manipulation"] = True
    if re.search(r"\bswim|pool|beach|water\b", q):
        f["swim"] = True
    if re.search(r"protective", q):
        f["protective"] = True

    for w, canon in LENGTH_WORDS.items():
        if w in q:
            f.setdefault("lengths", set()).add(canon)
    if "lengths" in f:
        f["lengths"] = sorted(f["lengths"])

    for w, canon in OCCASION_WORDS.items():
        if re.search(rf"\b{w}\b", q):
            f.setdefault("occasions", set()).add(canon)
    if "occasions" in f:
        f["occasions"] = sorted(f["occasions"])

    # numeric duration: "for 6 weeks", "lasts 2 months"
    m = re.search(r"(?:last|lasts|for)\s+(?:about\s+)?(\d+)\s*(week|weeks|month|months)", q)
    if m:
        n = int(m.group(1))
        f["min_weeks"] = n * 4 if "month" in m.group(2) else n

    # written-out numbers: "two months", "a couple of months", "next two months"
    _WORD_TO_N = {
        "a": 1, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8,
        "couple": 2, "few": 3, "several": 4,
    }
    m2 = re.search(
        r"(a\s+couple\s+of|a\s+few|several|a\s+|one|two|three|four|five|six|seven|eight)\s*"
        r"(week|weeks|month|months)",
        q,
    )
    if m2 and "min_weeks" not in f:
        raw = m2.group(1).strip().replace("a couple of", "couple").replace("a few", "few").replace("a", "a")
        # normalise multi-word tokens
        raw = re.sub(r"\s+", " ", raw).strip()
        word = raw.split()[-1]  # last token: "couple", "few", "two" etc.
        n = _WORD_TO_N.get(word, 1)
        f["min_weeks"] = n * 4 if "month" in m2.group(2) else n

    # vague long-duration intent: "a while", "ages", "long time", "won't have to think about"
    if "min_weeks" not in f and re.search(
        r"for a while|for ages|long time|long.lasting|don'?t want to (redo|redo)|"
        r"won'?t have to (think|touch|redo|worry)|traveling|on vacation|on holiday|"
        r"set (it )?and forget|won'?t (need to|have to) (do|touch|think)",
        q,
    ):
        f.setdefault("min_weeks", 6)  # implied: at least 6 weeks
        f.setdefault("low_manipulation", True)

    return f


# ---------------------------------------------------------------- embed backends

class FastembedBackend:
    name = "fastembed / BAAI bge-small-en-v1.5 (dense embeddings)"

    def __init__(self) -> None:
        from fastembed import TextEmbedding
        self._model = TextEmbedding("BAAI/bge-small-en-v1.5")

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self._model.embed(texts)))

    def encode_query(self, text: str) -> np.ndarray:
        return np.array(list(self._model.query_embed(text)))[0]


class TfidfSvdBackend:
    """Offline fallback: latent semantic vectors from TF-IDF + SVD."""

    name = "TF-IDF + SVD latent vectors (offline fallback)"

    def __init__(self, corpus: list[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        X = self._tfidf.fit_transform(corpus)
        k = min(64, X.shape[1] - 1, X.shape[0] - 1)
        self._svd = TruncatedSVD(n_components=k, random_state=0)
        self._svd.fit(X)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._svd.transform(self._tfidf.transform(texts))

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# ---------------------------------------------------------------- engine

def _norm(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class StyleSearch:
    W_DENSE = 0.45
    W_BM25 = 0.30
    W_FACET = 0.25

    def __init__(self) -> None:
        self.styles: list[dict] = json.loads(DATA.read_text())
        self.docs = [doc_text(s) for s in self.styles]
        self.bm25 = BM25Okapi([tokenize(d) for d in self.docs])
        try:
            self.backend = FastembedBackend()
        except Exception:
            self.backend = TfidfSvdBackend(self.docs)
        M = self.backend.encode(self.docs)
        self.doc_vecs = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)

    # -- facet scoring: domain knowledge, in code, not left to the model
    def _facet_score(self, s: dict, f: dict) -> tuple[float, list[str]]:
        score, hits, penalty = 0.0, [], 0.0

        if types := f.get("hair_types"):
            if any(t in s["hair_types"] for t in types):
                score += 1.0
                hits.append(f"suits {'/'.join(types)} hair")
            else:
                penalty += 1.5  # wrong hair type is near disqualifying

        if f.get("no_heat"):
            if not s["heat_required"]:
                score += 0.8
                hits.append("no heat")
            else:
                penalty += 2.0  # asked for no heat, style needs heat: hard penalty

        if f.get("low_manipulation"):
            if s["manipulation"] == "low":
                score += 0.8
                hits.append("low manipulation")
            elif s["manipulation"] == "high":
                penalty += 1.0

        if f.get("protective"):
            if s["protective"]:
                score += 0.6
                hits.append("protective")
            else:
                penalty += 1.0

        if f.get("swim"):
            if s.get("swim_friendly"):
                score += 0.8
                hits.append("swim friendly")
            else:
                penalty += 1.2

        if lengths := f.get("lengths"):
            if any(l in s["length_achieved"] for l in lengths):
                score += 0.6
                hits.append(f"works at {'/'.join(lengths)} length")
            else:
                penalty += 0.8

        if occs := f.get("occasions"):
            style_occs = set(s.get("occasions", []))
            matched = [o for o in occs if o in style_occs or (o == "interview" and "work" in style_occs)]
            if matched:
                score += 0.7
                hits.append(f"fits: {', '.join(matched)}")

        if wk := f.get("min_weeks"):
            if s["lasts_weeks"] >= wk:
                score += 0.8
                hits.append(f"lasts {s['lasts_weeks']} weeks")
            else:
                penalty += 1.2

        return score - penalty, hits

    def search(self, query: str, k: int = 8) -> dict:
        facets = extract_facets(query)

        qv = self.backend.encode_query(query)
        qv = qv / (np.linalg.norm(qv) + 1e-9)
        dense = self.doc_vecs @ qv

        bm = np.array(self.bm25.get_scores(tokenize(query)))

        facet_raw, facet_hits = [], []
        for s in self.styles:
            fs, hits = self._facet_score(s, facets)
            facet_raw.append(fs)
            facet_hits.append(hits)
        facet = np.array(facet_raw)

        dense_n, bm_n = _norm(dense), _norm(bm)
        # facets keep their sign: penalties must be able to sink a result
        facet_n = facet / (np.abs(facet).max() + 1e-9) if np.abs(facet).max() > 0 else facet

        final = self.W_DENSE * dense_n + self.W_BM25 * bm_n + self.W_FACET * facet_n
        order = np.argsort(-final)[:k]

        results = []
        for i in order:
            s = self.styles[int(i)]
            results.append({
                "style": s,
                "score": round(float(final[i]), 4),
                "signals": {
                    "dense": round(float(dense_n[i]), 3),
                    "bm25": round(float(bm_n[i]), 3),
                    "facets": round(float(facet_n[i]), 3),
                },
                "why": facet_hits[int(i)],
            })
        return {
            "query": query,
            "facets_detected": facets,
            "backend": self.backend.name,
            "results": results,
        }
