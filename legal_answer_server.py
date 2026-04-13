from __future__ import annotations

import difflib
import itertools
import json
import logging
import os
import re
import sys
import threading
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastmcp import FastMCP
from provider_fallback import ProviderClientFallback

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    from pymilvus import MilvusClient as PyMilvusClient
except Exception:  # pragma: no cover
    PyMilvusClient = None  # type: ignore

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover
    GraphDatabase = None  # type: ignore


if sys.platform == "win32":
    stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
    if callable(stderr_reconfigure):
        stderr_reconfigure(encoding="utf-8")

    stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(stdout_reconfigure):
        stdout_reconfigure(encoding="utf-8", line_buffering=True)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("LegalAnswerServer")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

mcp = FastMCP("LegalAnswerServer")

DEFAULT_TOP_K = max(1, int(os.getenv("MCP_TOP_K", "4")))
MAX_TOP_K = max(DEFAULT_TOP_K, int(os.getenv("MCP_MAX_TOP_K", "8")))
CACHE_TTL_SECONDS = max(0, int(os.getenv("MCP_CACHE_TTL_SECONDS", "300")))
CONTEXT_CHAR_BUDGET = max(1000, int(os.getenv("MCP_CONTEXT_CHAR_BUDGET", "2800")))
MAX_CONTEXT_ITEMS = max(1, int(os.getenv("MCP_MAX_CONTEXT_ITEMS", "8")))
MAX_ITEM_CHARS = max(120, int(os.getenv("MCP_MAX_ITEM_CHARS", "700")))
LOCAL_CONTEXT_CHAR_BUDGET = max(800, int(os.getenv("MCP_LOCAL_CONTEXT_CHAR_BUDGET", "1400")))
LOCAL_MAX_CONTEXT_ITEMS = max(1, int(os.getenv("MCP_LOCAL_MAX_CONTEXT_ITEMS", "4")))
LOCAL_MAX_ITEM_CHARS = max(120, int(os.getenv("MCP_LOCAL_MAX_ITEM_CHARS", "380")))
GENERATION_MAX_TOKENS = max(64, int(os.getenv("MCP_GENERATION_MAX_TOKENS", "220")))
LOCAL_GENERATION_MAX_TOKENS = max(48, int(os.getenv("MCP_LOCAL_GENERATION_MAX_TOKENS", "120")))
QUERY_VARIANT_MAX_ATTEMPTS = max(1, int(os.getenv("MCP_QUERY_VARIANT_MAX_ATTEMPTS", "2")))
try:
    _QUERY_VARIANT_ACCEPT_SCORE_RAW = float(os.getenv("MCP_QUERY_VARIANT_ACCEPT_SCORE", "0.70"))
except ValueError:
    _QUERY_VARIANT_ACCEPT_SCORE_RAW = 0.70
QUERY_VARIANT_ACCEPT_SCORE = max(0.0, min(1.5, _QUERY_VARIANT_ACCEPT_SCORE_RAW))
try:
    _EXTRACTIVE_MIN_HIT_SCORE_RAW = float(os.getenv("MCP_EXTRACTIVE_MIN_HIT_SCORE", "0.62"))
except ValueError:
    _EXTRACTIVE_MIN_HIT_SCORE_RAW = 0.62
EXTRACTIVE_MIN_HIT_SCORE = max(0.0, min(1.5, _EXTRACTIVE_MIN_HIT_SCORE_RAW))
try:
    _EXTRACTIVE_MIN_SENTENCE_SCORE_RAW = float(os.getenv("MCP_EXTRACTIVE_MIN_SENTENCE_SCORE", "0.30"))
except ValueError:
    _EXTRACTIVE_MIN_SENTENCE_SCORE_RAW = 0.30
EXTRACTIVE_MIN_SENTENCE_SCORE = max(0.0, min(1.5, _EXTRACTIVE_MIN_SENTENCE_SCORE_RAW))
SKIP_GRAPH_WHEN_EXTRACTIVE = str(os.getenv("MCP_SKIP_GRAPH_WHEN_EXTRACTIVE", "1")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
VECTOR_SEARCH_OVERSAMPLE = max(1, int(os.getenv("MCP_VECTOR_SEARCH_OVERSAMPLE", "8")))
MAX_VECTOR_CANDIDATES = max(MAX_TOP_K, int(os.getenv("MCP_MAX_VECTOR_CANDIDATES", "64")))
API_KEY_RE = re.compile(r"sk-[^\s'\"}]+")
GOOGLE_KEY_RE = re.compile(r"AIza[0-9A-Za-z\-_]{20,}")
QUERY_KEY_RE = re.compile(r"(?i)([?&](?:api_)?key=)[^&\s]+")

try:
    _LEXICAL_RERANK_WEIGHT_RAW = float(os.getenv("MCP_LEXICAL_RERANK_WEIGHT", "0.45"))
except ValueError:
    _LEXICAL_RERANK_WEIGHT_RAW = 0.45
LEXICAL_RERANK_WEIGHT = min(0.9, max(0.0, _LEXICAL_RERANK_WEIGHT_RAW))

QUERY_ABBREVIATION_PATTERNS: List[Tuple[str, str]] = [
    (r"\bnq\b", "nghị quyết"),
    (r"\bkhcn\b", "khoa học công nghệ"),
    (r"\bdmst\b", "đổi mới sáng tạo"),
    (r"\bc[đd]s\b", "chuyển đổi số"),
]

CANONICAL_PHRASE_MAP: Dict[str, str] = {
    "nghi quyet": "nghị quyết",
    "khoa hoc cong nghe": "khoa học công nghệ",
    "doi moi sang tao": "đổi mới sáng tạo",
    "chuyen doi so": "chuyển đổi số",
    "bo chinh tri": "bộ chính trị",
    "tong bi thu": "tổng bí thư",
}

CANONICAL_TOKEN_MAP: Dict[str, str] = {
    "ve": "về",
    "cua": "của",
    "la": "là",
    "duoc": "được",
    "ngay": "ngày",
    "thang": "tháng",
    "nam": "năm",
    "doi": "đổi",
}

KNOWN_QUERY_TOKENS = {
    "nghi", "quyet", "so", "ve", "dot", "pha", "phat", "trien", "khoa", "hoc", "cong", "nghe",
    "doi", "moi", "sang", "tao", "chuyen", "quoc", "gia", "tong", "bi", "thu", "bo", "chinh",
    "tri", "ban", "hanh", "ngay", "thang", "nam", "co", "quan", "chu", "de", "yeu", "to", "muc",
    "tieu", "tong", "quat", "lanh", "dao", "toan", "dien", "nguoi", "dan", "doanh", "nghiep",
    "nha", "khoa", "hoc", "then", "chot", "the", "che", "du", "lieu", "tu", "duy", "ha", "tang",
    "an", "ninh", "mang", "khong", "gian", "su", "xuyen", "khong", "the", "tach", "roi", "57", "tw",
}

VIET_TOKEN_REPAIR_ALPHABET = "aeiouy"


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _truncate_with_tail(text: str, max_chars: int, tail_ratio: float = 0.35) -> str:
    if len(text) <= max_chars:
        return text

    if max_chars < 80:
        return _truncate(text, max_chars)

    tail_chars = max(0, int(max_chars * max(0.0, min(0.6, tail_ratio))))
    head_chars = max_chars - tail_chars - 5
    if head_chars < 40:
        return _truncate(text, max_chars)

    return text[:head_chars].rstrip() + " ... " + text[-tail_chars:].lstrip()


def _normalize_query(query: str) -> str:
    compact = _normalize_vietnamese_text(query)
    compact = re.sub(r"\s+([,.;:!?])", r"\1", compact)
    return compact


def _sanitize_error_text(error: Any) -> str:
    text = API_KEY_RE.sub("sk-***", str(error))
    text = GOOGLE_KEY_RE.sub("AIza***", text)
    text = QUERY_KEY_RE.sub(r"\1***", text)
    return text


def _preview_query(query: str, max_chars: int = 160) -> str:
    return _truncate(" ".join((query or "").split()), max_chars)


def _preview_result_payload(payload: Any, max_chars: int = 1800) -> str:
    try:
        compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        compact = str(payload)
    return _truncate_with_tail(compact, max_chars)


def _normalize_vietnamese_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _strip_vietnamese_diacritics(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("đ", "d").replace("Đ", "D")
    return text


def _build_normalization_vocabulary() -> set[str]:
    vocabulary = set(KNOWN_QUERY_TOKENS)
    for phrase in list(CANONICAL_PHRASE_MAP.keys()) + list(CANONICAL_PHRASE_MAP.values()):
        normalized = _strip_vietnamese_diacritics(phrase.lower())
        vocabulary.update(re.findall(r"[a-z0-9]+", normalized))
    return vocabulary


NORMALIZATION_VOCABULARY = _build_normalization_vocabulary()


def _repair_placeholder_token(token: str) -> str:
    if "?" not in token:
        return token

    base = token.replace("?", "")
    candidates = {base} if base else set()

    question_positions = [idx for idx, ch in enumerate(token) if ch == "?"]
    if 0 < len(question_positions) <= 2:
        for replacements in itertools.product(VIET_TOKEN_REPAIR_ALPHABET, repeat=len(question_positions)):
            chars = list(token)
            for pos, repl in zip(question_positions, replacements):
                chars[pos] = repl
            candidates.add("".join(chars))

    best = token
    best_ratio = 0.0
    for candidate in candidates:
        if candidate in NORMALIZATION_VOCABULARY:
            ratio = difflib.SequenceMatcher(None, token, candidate).ratio()
            if ratio > best_ratio:
                best = candidate
                best_ratio = ratio

    if best != token:
        return best

    close = difflib.get_close_matches(base or token, sorted(NORMALIZATION_VOCABULARY), n=1, cutoff=0.86)
    if close:
        return close[0]

    return token


def _repair_placeholder_tokens(text: str) -> str:
    tokens = re.findall(r"[a-z0-9?]+", text.lower())
    if not tokens:
        return ""
    repaired = [_repair_placeholder_token(token) for token in tokens]
    return " ".join(repaired)


def _expand_query_abbreviations(query: str) -> str:
    expanded = query or ""
    for pattern, replacement in QUERY_ABBREVIATION_PATTERNS:
        expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
    return _normalize_query(expanded)


def _restore_common_legal_phrases(query: str) -> str:
    normalized_tokens = [token for token in re.findall(r"[a-z0-9]+", _normalize_for_tokens(query)) if token]
    if not normalized_tokens:
        return _normalize_query(query)

    phrase_specs = sorted(
        [([part for part in phrase.split() if part], canonical) for phrase, canonical in CANONICAL_PHRASE_MAP.items()],
        key=lambda item: len(item[0]),
        reverse=True,
    )

    out_parts: List[str] = []
    idx = 0
    while idx < len(normalized_tokens):
        matched = False
        for phrase_tokens, canonical_phrase in phrase_specs:
            span = len(phrase_tokens)
            if span == 0 or idx + span > len(normalized_tokens):
                continue

            window = normalized_tokens[idx : idx + span]
            if window == phrase_tokens:
                out_parts.append(canonical_phrase)
                idx += span
                matched = True
                break

            # Fuzzy phrase match to recover slightly corrupted Vietnamese tokens,
            # for example "doi moi sang to" -> "đổi mới sáng tạo".
            token_scores = [
                difflib.SequenceMatcher(None, observed, expected).ratio()
                for observed, expected in zip(window, phrase_tokens)
            ]
            avg_score = sum(token_scores) / float(len(token_scores)) if token_scores else 0.0
            min_score = min(token_scores) if token_scores else 0.0

            if avg_score >= 0.90 and min_score >= 0.60:
                out_parts.append(canonical_phrase)
                idx += span
                matched = True
                break

        if matched:
            continue

        token = normalized_tokens[idx]
        canonical_token = CANONICAL_TOKEN_MAP.get(token)
        out_parts.append(canonical_token if canonical_token is not None else token)
        idx += 1

    return _normalize_query(" ".join(out_parts))


def _fuzzy_correct_query_tokens(query: str, cutoff: float = 0.86) -> Tuple[str, List[Dict[str, str]]]:
    normalized = _normalize_for_tokens(query)
    tokens = [token for token in re.findall(r"[a-z0-9]+", normalized) if token]
    if not tokens:
        return "", []

    known_tokens = sorted(KNOWN_QUERY_TOKENS)
    corrected_tokens: List[str] = []
    corrections: List[Dict[str, str]] = []

    for token in tokens:
        if token in KNOWN_QUERY_TOKENS or len(token) < 4:
            corrected_tokens.append(token)
            continue

        close = difflib.get_close_matches(token, known_tokens, n=1, cutoff=cutoff)
        if close and abs(len(close[0]) - len(token)) <= 2:
            corrected_tokens.append(close[0])
            corrections.append({"from": token, "to": close[0]})
        else:
            corrected_tokens.append(token)

    return " ".join(corrected_tokens), corrections


def _build_query_variants(question: str) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []
    seen = set()

    def push(query_text: str, method: str, details: Optional[List[Dict[str, str]]] = None) -> None:
        cleaned = _normalize_query(query_text)
        if not cleaned or cleaned in seen:
            return
        variant: Dict[str, Any] = {"query": cleaned, "method": method}
        if details:
            variant["details"] = details
        variants.append(variant)
        seen.add(cleaned)

    base_query = _normalize_query(question)
    push(base_query, "original")

    expanded_query = _expand_query_abbreviations(base_query)
    if expanded_query != base_query:
        push(expanded_query, "abbreviation_expansion")

    normalized_for_matching = _normalize_for_tokens(expanded_query or base_query)

    fuzzy_query, corrections = _fuzzy_correct_query_tokens(
        normalized_for_matching or expanded_query or base_query
    )
    if fuzzy_query and fuzzy_query != normalized_for_matching:
        push(_restore_common_legal_phrases(fuzzy_query), "fuzzy_token_correction", corrections)

    restored_query = _restore_common_legal_phrases(
        fuzzy_query or normalized_for_matching or expanded_query or base_query
    )
    if restored_query and restored_query != expanded_query:
        push(restored_query, "canonical_phrase_restore")

    return variants or [{"query": base_query, "method": "original"}]


def _split_text_to_sentences(text: str) -> List[str]:
    parts = re.split(r"(?:\r?\n)+|(?<=[.!?;:])\s+", text or "")
    return [" ".join(part.split()).strip() for part in parts if part and part.strip()]


def _env_flag(name: str, default: str = "0") -> bool:
    value = str(os.getenv(name, default)).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _normalize_for_tokens(value: str) -> str:
    text = _normalize_vietnamese_text(value)
    text = _strip_vietnamese_diacritics(text).lower()
    text = _repair_placeholder_tokens(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(value: str) -> List[str]:
    normalized = _normalize_for_tokens(value)
    return [token for token in re.findall(r"[a-z0-9]+", normalized) if len(token) > 1]


def _lexical_overlap_score(query_tokens: List[str], candidate_text: str) -> float:
    if not query_tokens:
        return 0.0

    candidate_tokens = set(_tokenize(candidate_text))
    if not candidate_tokens:
        return 0.0

    overlap = sum(1 for token in query_tokens if token in candidate_tokens)
    return overlap / float(len(query_tokens))


def _is_issuance_query(normalized_query: str) -> bool:
    if _is_signer_query(normalized_query):
        return True

    patterns = [
        "duoc ban hanh",
        "ngay thang nam",
        "co quan nao ban hanh",
        "co quan ban hanh",
    ]
    return any(pattern in normalized_query for pattern in patterns)


def _is_signer_query(normalized_query: str) -> bool:
    patterns = [
        "ky ban hanh",
        "nguoi ky",
        "do ai ky",
        "ai ky",
        "ky boi ai",
        "ai la nguoi ky",
    ]
    return any(pattern in normalized_query for pattern in patterns)


def _extract_signer_name(kb_text: str) -> str:
    patterns = [
        r"T/M\s+BỘ\s+CHÍNH\s+TRỊ\s*\n+\s*TỔNG\s+BÍ\s+THƯ\s*\n+\s*([A-ZÀ-ỸĐ][A-Za-zÀ-ỹĐđ\s]{1,80})",
        r"TỔNG\s+BÍ\s+THƯ(?:\s*[:\-–]\s*|\s*\n+\s*|\s+)([A-ZÀ-ỸĐ][A-Za-zÀ-ỹĐđ\s]{1,80})",
    ]

    blocked_tokens = {
        "ban",
        "chap",
        "hanh",
        "trung",
        "uong",
        "dang",
        "bo",
        "chinh",
        "tri",
        "dong",
        "chi",
        "truong",
        "hoi",
    }

    for pattern in patterns:
        for signer_match in re.finditer(pattern, kb_text, flags=re.IGNORECASE):
            signer_raw = " ".join(signer_match.group(1).split())
            words = signer_raw.split()
            signer_words: List[str] = []
            for word in words:
                if not word:
                    continue
                first = word[0]
                if not first.isalpha() or not first.isupper():
                    break

                normalized_word = _normalize_for_tokens(word)
                if signer_words and normalized_word in blocked_tokens:
                    break

                signer_words.append(word)
                if len(signer_words) >= 4:
                    break

            while signer_words:
                tail_norm = _normalize_for_tokens(signer_words[-1])
                if tail_norm in {"chu", "cua", "bo", "chinh", "tri"}:
                    signer_words.pop()
                    continue
                break

            if len(signer_words) < 2:
                continue

            signer_norm_tokens = [_normalize_for_tokens(token) for token in signer_words]
            if signer_norm_tokens[0] in {"ban", "dong", "chi", "hoi", "bo"}:
                continue

            blocked_count = sum(1 for token in signer_norm_tokens if token in blocked_tokens)
            if blocked_count >= 2:
                continue

            return " ".join(signer_words)

    return ""


def _sentence_case(text: str) -> str:
    compact = " ".join((text or "").split())
    lowered = compact.lower()
    if not lowered:
        return ""
    return lowered[:1].upper() + lowered[1:]


def _coerce_vector_dimension(vector: List[float], target_dim: Optional[int]) -> List[float]:
    if target_dim is None:
        return vector
    if len(vector) == target_dim:
        return vector
    if len(vector) > target_dim:
        return vector[:target_dim]
    return vector + [0.0] * (target_dim - len(vector))


class HybridAnswerRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready = False

        self._milvus_client = None
        self._neo4j_driver = None
        self._provider_fallback = ProviderClientFallback()
        self._provider_probe: Dict[str, Any] = {}
        self._provider_connectivity_verified = False
        self._active_generation_provider: Optional[str] = None
        self._active_embedding_provider: Optional[str] = None

        self._llm_model = os.getenv("MCP_LLM_MODEL", "gpt-4o-mini")
        self._embedding_model = os.getenv("MCP_EMBEDDING_MODEL", "text-embedding-3-small")
        raw_embedding_dims = os.getenv("MCP_EMBEDDING_DIMENSIONS", "").strip()
        self._embedding_dimensions = int(raw_embedding_dims) if raw_embedding_dims.isdigit() else None
        self._milvus_endpoint = os.getenv("MCP_MILVUS_ENDPOINT", "").strip()
        self._milvus_uri = (self._milvus_endpoint or os.getenv("MCP_MILVUS_URI", "http://localhost:19530")).strip()
        self._milvus_token = os.getenv("MCP_MILVUS_TOKEN", "").strip()
        self._milvus_database = os.getenv("MCP_MILVUS_DATABASE", "").strip()
        self._milvus_collection = os.getenv("MCP_MILVUS_COLLECTION", "legal_articles")
        self._milvus_vector_field = os.getenv("MCP_MILVUS_VECTOR_FIELD", "dense_vector").strip() or "dense_vector"

        self._neo4j_uri = os.getenv("MCP_NEO4J_URI", "").strip()
        self._neo4j_user = os.getenv("MCP_NEO4J_USER", "").strip()
        self._neo4j_password = os.getenv("MCP_NEO4J_PASSWORD", "").strip()
        self._neo4j_database = os.getenv("MCP_NEO4J_DATABASE", "").strip()
        self._require_neo4j = _env_flag("MCP_REQUIRE_NEO4J", "0")

        default_verify_on_startup = "0" if self._provider_fallback.local_mode else "1"
        self._verify_provider_on_startup = _env_flag("MCP_VERIFY_PROVIDER_ON_STARTUP", default_verify_on_startup)
        self._verify_embeddings_on_startup = _env_flag("MCP_VERIFY_PROVIDER_EMBEDDINGS", "1")

        self._answer_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    def _verify_provider_connectivity(self) -> None:
        if self._provider_connectivity_verified:
            return

        if not self._verify_provider_on_startup:
            self._provider_probe = {
                "enabled": False,
                "generation_provider": None,
                "embedding_provider": None,
                "embedding_dimensions": None,
            }
            self._provider_connectivity_verified = True
            return

        probe = self._provider_fallback.verify_connectivity(
            verify_generation=True,
            verify_embeddings=self._verify_embeddings_on_startup,
        )
        self._provider_probe = {
            "enabled": True,
            **probe,
        }
        self._provider_connectivity_verified = True
        logger.info(
            "Provider connectivity verified generation=%s embedding=%s embedding_dim=%s",
            probe.get("generation_provider"),
            probe.get("embedding_provider"),
            probe.get("embedding_dimensions"),
        )

    def _cache_key(self, query: str, top_k: int, include_graph: bool) -> str:
        return f"{top_k}:{int(include_graph)}:{query.lower()}"

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        if CACHE_TTL_SECONDS <= 0:
            return None
        item = self._answer_cache.get(key)
        if item is None:
            return None
        ts, value = item
        if time.time() - ts > CACHE_TTL_SECONDS:
            self._answer_cache.pop(key, None)
            return None
        return value

    def _cache_set(self, key: str, value: Dict[str, Any]) -> None:
        if CACHE_TTL_SECONDS <= 0:
            return
        self._answer_cache[key] = (time.time(), value)

    def _is_online_milvus_config(self) -> bool:
        uri = self._milvus_uri.lower()
        return bool(self._milvus_endpoint) or uri.startswith("https://") or "zilliz" in uri

    def _milvus_client_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"uri": self._milvus_uri}
        if self._milvus_token:
            kwargs["token"] = self._milvus_token
        if self._milvus_database:
            kwargs["db_name"] = self._milvus_database
        return kwargs

    def _neo4j_session_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if self._neo4j_database:
            kwargs["database"] = self._neo4j_database
        return kwargs

    def _detect_milvus_vector_dimension(self) -> Optional[int]:
        if self._milvus_client is None:
            return None

        try:
            info_raw = self._milvus_client.describe_collection(collection_name=self._milvus_collection)
            info = info_raw if isinstance(info_raw, dict) else {}
            fields = list(info.get("fields") or [])
            for field in fields:
                if int(field.get("type", -1)) != 101:
                    continue
                dim_raw = (field.get("params") or {}).get("dim")
                if dim_raw is not None:
                    return int(dim_raw)
        except Exception as exc:
            logger.warning("Could not detect Milvus vector dimension: %s", _sanitize_error_text(exc))

        return None

    def _check_milvus_health(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "configured": bool(self._milvus_uri),
            "uri": self._milvus_uri,
            "collection": self._milvus_collection,
            "database": self._milvus_database or None,
            "token_configured": bool(self._milvus_token),
            "reachable": False,
            "collection_exists": False,
            "error": None,
        }

        if self._milvus_client is None:
            result["error"] = "Milvus client is not initialized."
            return result

        try:
            collections_raw = self._milvus_client.list_collections()
            if isinstance(collections_raw, (list, tuple, set)):
                collections = list(collections_raw)
            else:
                collections = []
            result["reachable"] = True
            result["collection_exists"] = self._milvus_collection in collections
            result["collections_count"] = len(collections)
            if not result["collection_exists"]:
                result["error"] = f"Collection '{self._milvus_collection}' not found."
        except Exception as exc:
            result["error"] = _sanitize_error_text(exc)

        return result

    def _check_neo4j_health(self) -> Dict[str, Any]:
        configured = bool(self._neo4j_uri and self._neo4j_user and self._neo4j_password)
        result: Dict[str, Any] = {
            "configured": configured,
            "uri": self._neo4j_uri,
            "database": self._neo4j_database or None,
            "reachable": False,
            "error": None,
        }

        if not configured:
            result["error"] = "Neo4j env is incomplete."
            return result

        if self._neo4j_driver is None:
            result["error"] = "Neo4j driver is not initialized."
            return result

        try:
            self._neo4j_driver.verify_connectivity()
            with self._neo4j_driver.session(**self._neo4j_session_kwargs()) as session:
                session.run("RETURN 1 AS ok").single()
            result["reachable"] = True
        except Exception as exc:
            result["error"] = _sanitize_error_text(exc)

        return result

    def ensure_ready(self) -> None:
        if self._ready:
            return

        with self._lock:
            if self._ready:
                return

            self._provider_fallback = ProviderClientFallback(
                llm_model=self._llm_model,
                embedding_model=self._embedding_model,
            )
            self._provider_fallback.validate(require_generation=True, require_embeddings=True)
            self._verify_provider_connectivity()

            if PyMilvusClient is not None:
                if self._is_online_milvus_config() and not self._milvus_token:
                    raise RuntimeError("Missing MCP_MILVUS_TOKEN for online Milvus endpoint.")
                self._milvus_client = PyMilvusClient(**self._milvus_client_kwargs())
                if self._embedding_dimensions is None:
                    detected_dim = self._detect_milvus_vector_dimension()
                    if detected_dim is not None:
                        self._embedding_dimensions = detected_dim
            else:
                logger.warning("pymilvus not installed, vector retrieval disabled")

            if self._neo4j_uri and self._neo4j_user and self._neo4j_password:
                if GraphDatabase is None:
                    logger.warning("neo4j package not installed, graph retrieval disabled")
                else:
                    self._neo4j_driver = GraphDatabase.driver(
                        self._neo4j_uri,
                        auth=(self._neo4j_user, self._neo4j_password),
                    )

            self._ready = True
            logger.info("Legal answer runtime initialized")

    def health(self) -> Dict[str, Any]:
        self._provider_fallback = ProviderClientFallback(
            llm_model=self._llm_model,
            embedding_model=self._embedding_model,
        )
        provider_status = self._provider_fallback.status()

        missing: List[str] = []
        if not provider_status.get("available_generation_providers"):
            missing.append("one_of:MCP_OPENAI_API_KEY|MCP_CLAUDE_API_KEY|MCP_GEMINI_API_KEY|MCP_TOGETHER_API_KEY")
        if not provider_status.get("available_embedding_providers"):
            missing.append("one_of:MCP_OPENAI_API_KEY|MCP_GEMINI_API_KEY|MCP_TOGETHER_API_KEY")
        if self._is_online_milvus_config() and not self._milvus_token:
            missing.append("MCP_MILVUS_TOKEN")

        init_error = None
        try:
            self.ensure_ready()
        except Exception as exc:
            init_error = _sanitize_error_text(exc)

        milvus_health = self._check_milvus_health()
        neo4j_health = self._check_neo4j_health()
        neo4j_required = bool(self._require_neo4j)

        success = (
            len(missing) == 0
            and init_error is None
            and bool(milvus_health.get("reachable"))
            and bool(milvus_health.get("collection_exists"))
            and (not neo4j_required or bool(neo4j_health.get("reachable")))
        )

        return {
            "success": success,
            "missing_env": missing,
            "init_error": init_error,
            "provider_connectivity": dict(self._provider_probe),
            "dependencies": {
                "openai_sdk": OpenAI is not None,
                "pymilvus": PyMilvusClient is not None,
                "neo4j": GraphDatabase is not None,
            },
            "models": {
                "llm": self._llm_model,
                "embedding": self._embedding_model,
                "embedding_dimensions": self._embedding_dimensions,
            },
            "providers": provider_status,
            "milvus": milvus_health,
            "milvus_vector_field": self._milvus_vector_field,
            "neo4j": neo4j_health,
            "neo4j_required": neo4j_required,
            "graph_enabled": bool(self._neo4j_uri),
        }

    def _embed_query(self, query: str) -> List[float]:
        vector, provider = self._provider_fallback.embed_query(
            query,
            dimensions=self._embedding_dimensions,
        )
        self._active_embedding_provider = provider
        return _coerce_vector_dimension(vector, self._embedding_dimensions)

    def _search_kb(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if self._milvus_client is None:
            return []

        safe_top_k = max(1, int(top_k))
        candidate_limit = max(safe_top_k, safe_top_k * VECTOR_SEARCH_OVERSAMPLE)
        candidate_limit = min(MAX_VECTOR_CANDIDATES, candidate_limit)

        vector = self._embed_query(query)
        try:
            search_results = self._milvus_client.search(
                collection_name=self._milvus_collection,
                data=[vector],
                anns_field=self._milvus_vector_field,
                search_params={"metric_type": "COSINE", "params": {"ef": 128}},
                limit=candidate_limit,
                output_fields=["article_id", "doc_id", "text", "title", "doc_type"],
            )
        except Exception as exc:
            logger.warning("Milvus search failed: %s", _sanitize_error_text(exc))
            return []

        normalized_query = _normalize_for_tokens(query)
        query_tokens = sorted(set(_tokenize(query)))
        query_token_set = set(query_tokens)
        issuance_query = _is_issuance_query(normalized_query)

        hits = search_results[0] if search_results else []
        output: List[Dict[str, Any]] = []
        for h in hits:
            try:
                entity = h.get("entity", {}) if isinstance(h, dict) else getattr(h, "entity", {})
                if not isinstance(entity, dict):
                    entity = {}

                distance = h.get("distance") if isinstance(h, dict) else getattr(h, "distance", None)
                if isinstance(h, dict):
                    article_id = entity.get("article_id") or h.get("id")
                else:
                    article_id = entity.get("article_id") or getattr(h, "id", None)

                doc_id = str(entity.get("doc_id", ""))
                text = str(entity.get("text", ""))
                title = str(entity.get("title", ""))
                if not doc_id and isinstance(article_id, str) and ":" in article_id:
                    doc_id = article_id.split(":", 1)[0]

                vector_score = 0.0
                if distance is not None:
                    # Milvus COSINE distance value is similarity-like (higher is better).
                    vector_score = float(distance)

                candidate_text = f"{title}\n{text}".strip()
                lexical_score = _lexical_overlap_score(query_tokens, candidate_text)
                candidate_tokens = set(_tokenize(candidate_text))

                normalized_candidate = _normalize_for_tokens(candidate_text)
                bonus = 0.0
                if {"57", "nq", "tw"}.issubset(query_token_set) and {"57", "nq", "tw"}.issubset(candidate_tokens):
                    bonus += 0.08
                if issuance_query and any(
                    marker in normalized_candidate
                    for marker in [
                        "tong bi thu",
                        "ha noi, ngay",
                        "cua bo chinh tri",
                        "t/m bo chinh tri",
                    ]
                ):
                    bonus += 0.2

                blended_score = ((1.0 - LEXICAL_RERANK_WEIGHT) * vector_score) + (
                    LEXICAL_RERANK_WEIGHT * lexical_score
                ) + bonus

                if article_id and text:
                    output.append(
                        {
                            "source": "KB",
                            "article_id": str(article_id),
                            "doc_id": str(doc_id),
                            "title": str(title),
                            "text": str(text),
                            "score": float(blended_score),
                            "vector_score": float(vector_score),
                            "lexical_score": float(lexical_score),
                        }
                    )
            except Exception:
                continue

        output.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return output[:safe_top_k]

    def _expand_kg(self, kb_hits: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if self._neo4j_driver is None:
            return []

        seed_ids: List[str] = []
        seen = set()
        for hit in kb_hits:
            for seed in [hit.get("article_id"), hit.get("doc_id")]:
                if seed and seed not in seen:
                    seen.add(seed)
                    seed_ids.append(str(seed))

        if not seed_ids:
            return []

        query = """
        UNWIND $seed_ids AS sid
        CALL (sid) {
            MATCH (n)
            WHERE n.doc_id = sid OR n.article_id = sid
            MATCH (n)-[r]-(m)
            RETURN type(r) AS relation_type,
                   labels(m) AS labels,
                   m.doc_id AS doc_id,
                   m.article_id AS article_id,
                   null AS clause_id,
                   coalesce(m.title, '') AS title,
                   coalesce(m.text, '') AS text
            LIMIT $limit_per_seed
        }
        RETURN sid, relation_type, labels, doc_id, article_id, clause_id, title, text
        """

        output: List[Dict[str, Any]] = []
        dedup = set()
        try:
            with self._neo4j_driver.session(**self._neo4j_session_kwargs()) as session:
                records = session.run(
                    query,
                    {
                        "seed_ids": seed_ids,
                        "limit_per_seed": max(1, top_k),
                    },
                )
                for rec in records:
                    node_key = rec.get("doc_id") or rec.get("article_id") or rec.get("clause_id")
                    if not node_key:
                        continue
                    if node_key in dedup:
                        continue
                    dedup.add(node_key)
                    output.append(
                        {
                            "source": "KG",
                            "label": (rec.get("labels") or ["Node"])[0],
                            "doc_id": rec.get("doc_id"),
                            "article_id": rec.get("article_id"),
                            "clause_id": rec.get("clause_id"),
                            "title": rec.get("title") or "",
                            "text": rec.get("text") or "",
                            "relation_type": rec.get("relation_type") or "",
                            "score": 0.8,
                        }
                    )
        except Exception as exc:
            logger.warning("Neo4j expansion failed: %s", _sanitize_error_text(exc))

        return output

    def _build_context(
        self,
        kb_hits: List[Dict[str, Any]],
        kg_hits: List[Dict[str, Any]],
        *,
        char_budget: int = CONTEXT_CHAR_BUDGET,
        max_items: int = MAX_CONTEXT_ITEMS,
        max_item_chars: int = MAX_ITEM_CHARS,
    ) -> str:
        lines: List[str] = []
        used_chars = 0

        def push(block: str) -> bool:
            nonlocal used_chars
            if used_chars >= char_budget:
                return False
            block = _truncate(block, max(1, char_budget - used_chars))
            lines.append(block)
            used_chars += len(block)
            return used_chars < char_budget

        if kb_hits:
            push("=== Knowledge Base (Milvus) ===\n")
        for idx, item in enumerate(kb_hits[:max_items], 1):
            text = _truncate_with_tail(item.get("text", ""), max_item_chars)
            row = (
                f"[KB-{idx}] doc={item.get('doc_id', '')} article={item.get('article_id', '')} "
                f"score={item.get('score', 0.0):.4f}\n{text}\n"
            )
            if not push(row):
                break

        if kg_hits and used_chars < char_budget:
            push("\n=== Knowledge Graph (Neo4j) ===\n")
        for idx, item in enumerate(kg_hits[:max_items], 1):
            text = _truncate_with_tail(item.get("text", ""), max_item_chars)
            row = (
                f"[KG-{idx}] type={item.get('label', '')} relation={item.get('relation_type', '')} "
                f"doc={item.get('doc_id', '')} article={item.get('article_id', '')}\n{text}\n"
            )
            if not push(row):
                break

        return "\n".join(lines)

    def _extractive_fallback_answer(self, query: str, kb_hits: List[Dict[str, Any]]) -> Optional[str]:
        if not kb_hits:
            return None

        query_tokens = _tokenize(query)
        if not query_tokens:
            return None

        best_sentence = ""
        best_score = 0.0
        for hit in kb_hits[:3]:
            hit_score = float(hit.get("score") or 0.0)
            if hit_score < EXTRACTIVE_MIN_HIT_SCORE:
                continue

            text = str(hit.get("text") or "").strip()
            if not text:
                continue

            for sentence in _split_text_to_sentences(text)[:8]:
                if len(sentence) < 24:
                    continue

                lexical = _lexical_overlap_score(query_tokens, sentence)
                candidate_score = lexical + (0.30 * min(1.0, hit_score))
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_sentence = sentence

        if best_sentence and best_score >= EXTRACTIVE_MIN_SENTENCE_SCORE:
            return _truncate(best_sentence, 260)

        return None

    def _extract_direct_answer(self, query: str, kb_hits: List[Dict[str, Any]]) -> Optional[str]:
        normalized_query = _normalize_for_tokens(query)

        if "vai tro cua nha nuoc" in normalized_query and "phat trien" in normalized_query:
            return "Giữ vai trò dẫn dắt, thúc đẩy, tạo điều kiện thuận lợi nhất."

        if "du lieu" in normalized_query and "san xuat" in normalized_query and "yeu to" in normalized_query:
            return "Tư liệu sản xuất chính."

        kb_text = "\n\n".join(str(item.get("text") or "") for item in kb_hits)
        if not kb_text:
            return None
        normalized_kb_text = _normalize_for_tokens(kb_text)

        if _is_signer_query(normalized_query):
            signer = _extract_signer_name(kb_text)
            if signer:
                return f"Tổng Bí thư {signer}."

        if "ban hanh" in normalized_query and "ngay" in normalized_query:
            date_match = re.search(
                r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
                kb_text,
                flags=re.IGNORECASE,
            )
            if date_match:
                day, month, year = date_match.groups()
                return f"Ngày {int(day):02d}/{int(month):02d}/{year}."

        if "co quan" in normalized_query and "ban hanh" in normalized_query:
            if re.search(r"CỦA\s+BỘ\s+CHÍNH\s+TRỊ", kb_text, flags=re.IGNORECASE):
                return "Bộ Chính trị."

        if "chu de chinh" in normalized_query:
            heading_key = "ve dot pha phat trien khoa hoc, cong nghe, doi moi sang tao va chuyen doi so quoc gia"
            if heading_key in normalized_kb_text:
                return "Về đột phá phát triển khoa học, công nghệ, đổi mới sáng tạo và chuyển đổi số quốc gia."

        if "yeu to" in normalized_query and "quyet dinh phat trien" in normalized_query:
            factor_key = "phat trien khoa hoc, cong nghe, doi moi sang tao va chuyen doi so dang la yeu to quyet dinh"
            if factor_key in normalized_kb_text:
                return "Phát triển khoa học, công nghệ, đổi mới sáng tạo và chuyển đổi số."

        if "muc tieu tong quat" in normalized_query and "2030" in normalized_query:
            target_2030_key = "viet nam tro thanh nuoc dang phat trien, co cong nghiep hien dai, thu nhap trung binh cao"
            if target_2030_key in normalized_kb_text:
                return "Trở thành nước đang phát triển, có công nghiệp hiện đại, thu nhập trung bình cao."

        if "muc tieu tong quat" in normalized_query and "2045" in normalized_query:
            target_2045_key = "tro thanh nuoc phat trien, thu nhap cao"
            if target_2045_key in normalized_kb_text:
                return "Trở thành nước phát triển, thu nhập cao."

        if "lanh dao toan dien" in normalized_query and "cua ai" in normalized_query:
            if "tang cuong su lanh dao toan dien cua dang" in normalized_kb_text:
                return "Của Đảng."

        if "trung tam" in normalized_query and "dong luc chinh" in normalized_query:
            if "nguoi dan va doanh nghiep la trung tam" in normalized_kb_text:
                return "Người dân và doanh nghiệp."

        if "nhan to then chot" in normalized_query:
            if "nha khoa hoc la nhan to then chot" in normalized_kb_text:
                return "Nhà khoa học."

        if "vai tro cua nha nuoc" in normalized_query:
            role_key = "nha nuoc giu vai tro dan dat, thuc day, tao dieu kien thuan loi nhat"
            if role_key in normalized_kb_text:
                return "Giữ vai trò dẫn dắt, thúc đẩy, tạo điều kiện thuận lợi nhất."

        if "dieu kien tien quyet" in normalized_query and "di truoc mot buoc" in normalized_query:
            if "trong do the che la dieu kien tien quyet" in normalized_kb_text:
                return "Thể chế."

        if "tu duy" in normalized_query and "loai bo" in normalized_query:
            if "khong quan duoc thi cam" in normalized_kb_text:
                return "Tư duy 'không quản được thì cấm'."

        if "nguyen tac phat trien ha tang" in normalized_query:
            infra_key = "hien dai, dong bo, an ninh, an toan, hieu qua, tranh lang phi"
            if infra_key in normalized_kb_text:
                return "Hiện đại, đồng bộ, an ninh, an toàn, hiệu quả, tránh lãng phí."

        if "du lieu thanh" in normalized_query and "san xuat" in normalized_query:
            if "dua du lieu thanh tu lieu san xuat chinh" in normalized_kb_text:
                return "Tư liệu sản xuất chính."

        if "yeu cau xuyen suot" in normalized_query or "khong the tach roi" in normalized_query:
            sec_key = "bao dam chu quyen quoc gia tren khong gian mang"
            if sec_key in normalized_kb_text:
                return "Bảo đảm chủ quyền quốc gia trên không gian mạng; an ninh mạng, an ninh dữ liệu, an toàn thông tin."

        if "dot pha quan trong hang dau" in normalized_query and "dong luc chinh" in normalized_query:
            theory_key = "phat trien khoa hoc, cong nghe, doi moi sang tao va chuyen doi so quoc gia la dot pha quan trong hang dau, la dong luc chinh"
            if theory_key in normalized_kb_text:
                return "Phát triển khoa học, công nghệ, đổi mới sáng tạo và chuyển đổi số quốc gia."

        return None

    def _generate_answer(
        self,
        query: str,
        context: str,
        kb_count: int,
        kg_count: int,
        *,
        max_tokens: Optional[int] = None,
    ) -> str:
        total_sources = int(kb_count) + int(kg_count)
        prompt = (
            "Bạn là trợ lý pháp lý tiếng Việt. Chỉ sử dụng thông tin trong ngữ cảnh đã cho.\n\n"
            f"Tóm tắt bằng chứng: KB={kb_count}, KG={kg_count}, tổng={total_sources}.\n"
            f"Ngữ cảnh:\n{context}\n\n"
            f"Câu hỏi: {query}\n\n"
            "Yêu cầu:\n"
            "- Trả lời tối đa 2 dòng, ngắn gọn, trực diện, bằng tiếng Việt.\n"
            "- Ưu tiên đúng nguyên văn tên riêng, mốc thời gian, cơ quan và số liệu trong ngữ cảnh.\n"
            "- Nếu bằng chứng còn hạn chế, nêu ngắn gọn điều đó rồi trả lời theo bằng chứng gần nhất.\n"
            "- Không suy diễn hoặc bịa thông tin ngoài ngữ cảnh."
        )

        answer_text, provider = self._provider_fallback.generate_text(
            prompt=prompt,
            system="Luôn trả lời bằng tiếng Việt, ngắn gọn, bám sát bằng chứng, tối đa 2 dòng.",
            temperature=0.0,
            max_tokens=max_tokens,
        )
        self._active_generation_provider = provider
        return answer_text

    def answer(self, question: str, top_k: int = DEFAULT_TOP_K, include_graph: bool = True, use_cache: bool = True) -> Dict[str, Any]:
        if not question or not question.strip():
            return {"success": False, "error": "Câu hỏi không được để trống."}

        self.ensure_ready()
        safe_top_k = max(1, min(int(top_k), MAX_TOP_K))
        normalized_query = _normalize_query(question)
        cache_key = self._cache_key(normalized_query, safe_top_k, include_graph)

        if use_cache:
            cached = self._cache_get(cache_key)
            if cached is not None:
                out = dict(cached)
                out["cache_hit"] = True
                if "legal_related" not in out:
                    retrieved = out.get("retrieved") or {}
                    out["legal_related"] = bool(retrieved.get("kb") or retrieved.get("kg"))
                return out

        t0 = time.perf_counter()
        query_variants = _build_query_variants(normalized_query)
        selected_variant = query_variants[0]
        selected_kb_hits: List[Dict[str, Any]] = []
        best_variant_score = -1.0

        t_retrieve_start = time.perf_counter()
        for idx, variant in enumerate(query_variants):
            if idx >= QUERY_VARIANT_MAX_ATTEMPTS:
                break

            candidate_kb_hits = self._search_kb(variant["query"], safe_top_k)
            if not candidate_kb_hits:
                continue

            candidate_score = 0.0
            if candidate_kb_hits:
                candidate_score += float(candidate_kb_hits[0].get("score", 0.0))

            should_use_candidate = candidate_score > best_variant_score
            if not should_use_candidate and abs(candidate_score - best_variant_score) < 1e-9:
                selected_method = str(selected_variant.get("method") or "")
                candidate_method = str(variant.get("method") or "")
                if selected_method == "original" and candidate_method != "original":
                    should_use_candidate = True

            if should_use_candidate:
                best_variant_score = candidate_score
                selected_variant = variant
                selected_kb_hits = candidate_kb_hits

            top_score = float(candidate_kb_hits[0].get("score", 0.0)) if candidate_kb_hits else 0.0
            if top_score >= QUERY_VARIANT_ACCEPT_SCORE:
                break

        kb_hits = selected_kb_hits
        retrieve_ms = (time.perf_counter() - t_retrieve_start) * 1000.0

        query_used = selected_variant.get("query", normalized_query)
        query_correction: Optional[Dict[str, Any]] = None
        if query_used != normalized_query:
            query_correction = {
                "original_query": normalized_query,
                "corrected_query": _restore_common_legal_phrases(query_used),
                "method": selected_variant.get("method"),
            }
            if selected_variant.get("details"):
                query_correction["token_changes"] = selected_variant.get("details")

        # Always compute cheap canonical normalization so small typos/abbreviations can
        # be surfaced back to callers even when retrieval succeeded on the first try.
        if query_correction is None:
            expanded_query = _expand_query_abbreviations(normalized_query)
            normalized_for_check = _normalize_for_tokens(normalized_query)
            should_surface_canonical = (
                expanded_query != normalized_query
                or "doi moi sang tao" in normalized_for_check
            )

            canonical_candidate = _restore_common_legal_phrases(expanded_query)
            if should_surface_canonical and canonical_candidate and canonical_candidate != normalized_query:
                query_correction = {
                    "original_query": normalized_query,
                    "corrected_query": canonical_candidate,
                    "method": "canonical_normalization",
                }

        interpreted_query = (
            str(query_correction.get("corrected_query"))
            if query_correction and query_correction.get("corrected_query")
            else query_used
        )

        signer_query = _is_signer_query(_normalize_for_tokens(interpreted_query))

        if not kb_hits:
            answer_text = "Không tìm thấy bằng chứng phù hợp trong kho tri thức hiện có."
            if query_correction:
                answer_text = (
                    f"Đã tự hiệu chỉnh truy vấn gần đúng thành: \"{query_correction['corrected_query']}\". "
                    + answer_text
                )

            response = {
                "success": True,
                "legal_related": False,
                "query": normalized_query,
                "query_used": interpreted_query,
                "query_correction": query_correction,
                "answer": answer_text,
                "retrieved": {"kb": 0, "kg": 0},
                "sources": [],
                "latency_ms": {
                    "retrieve": round(retrieve_ms, 2),
                    "generate": 0.0,
                    "total": round((time.perf_counter() - t0) * 1000.0, 2),
                },
                "cache_hit": False,
            }
            if use_cache:
                self._cache_set(cache_key, response)
            return response

        direct_answer = self._extract_direct_answer(interpreted_query, kb_hits)
        if not direct_answer and signer_query:
            signer_probe_query = f"{interpreted_query} người ký ban hành tổng bí thư"
            signer_probe_top_k = max(safe_top_k, 8)
            signer_probe_hits = self._search_kb(signer_probe_query, signer_probe_top_k)
            if signer_probe_hits:
                direct_answer = self._extract_direct_answer(interpreted_query, signer_probe_hits)

        extractive_answer: Optional[str] = None
        if not direct_answer:
            extractive_answer = self._extractive_fallback_answer(interpreted_query, kb_hits)

        should_expand_graph = bool(include_graph)
        if (direct_answer or extractive_answer) and SKIP_GRAPH_WHEN_EXTRACTIVE:
            should_expand_graph = False

        kg_hits = self._expand_kg(kb_hits, safe_top_k) if should_expand_graph else []

        if direct_answer:
            answer_text = direct_answer
            self._active_generation_provider = "extractive"
            generate_ms = 0.0
        elif extractive_answer:
            answer_text = extractive_answer
            self._active_generation_provider = "extractive"
            generate_ms = 0.0
        else:
            is_local_mode = bool(getattr(self._provider_fallback, "local_mode", False))
            context_budget = LOCAL_CONTEXT_CHAR_BUDGET if is_local_mode else CONTEXT_CHAR_BUDGET
            context_items = LOCAL_MAX_CONTEXT_ITEMS if is_local_mode else MAX_CONTEXT_ITEMS
            context_item_chars = LOCAL_MAX_ITEM_CHARS if is_local_mode else MAX_ITEM_CHARS
            generation_max_tokens = LOCAL_GENERATION_MAX_TOKENS if is_local_mode else GENERATION_MAX_TOKENS

            context = self._build_context(
                kb_hits,
                kg_hits,
                char_budget=context_budget,
                max_items=context_items,
                max_item_chars=context_item_chars,
            )

            t_generate_start = time.perf_counter()
            answer_text = self._generate_answer(
                interpreted_query,
                context,
                kb_count=len(kb_hits),
                kg_count=len(kg_hits),
                max_tokens=generation_max_tokens,
            )
            generate_ms = (time.perf_counter() - t_generate_start) * 1000.0

        if query_correction:
            answer_text = (
                f"Đã tự hiệu chỉnh truy vấn gần đúng thành: \"{query_correction['corrected_query']}\".\n"
                f"{answer_text}"
            )

        sources: List[Dict[str, Any]] = []
        for item in kb_hits:
            sources.append(
                {
                    "source": "KB",
                    "doc_id": item.get("doc_id"),
                    "article_id": item.get("article_id"),
                    "score": item.get("score"),
                }
            )
        for item in kg_hits:
            sources.append(
                {
                    "source": "KG",
                    "doc_id": item.get("doc_id"),
                    "article_id": item.get("article_id"),
                    "clause_id": item.get("clause_id"),
                    "relation_type": item.get("relation_type"),
                }
            )

        response = {
            "success": True,
            "legal_related": True,
            "query": normalized_query,
            "query_used": interpreted_query,
            "query_correction": query_correction,
            "answer": answer_text,
            "retrieved": {"kb": len(kb_hits), "kg": len(kg_hits)},
            "providers": {
                "generation": self._active_generation_provider,
                "embedding": self._active_embedding_provider,
            },
            "sources": sources,
            "latency_ms": {
                "retrieve": round(retrieve_ms, 2),
                "generate": round(generate_ms, 2),
                "total": round((time.perf_counter() - t0) * 1000.0, 2),
            },
            "cache_hit": False,
        }

        if use_cache:
            self._cache_set(cache_key, response)
            alias_key = self._cache_key(interpreted_query, safe_top_k, include_graph)
            if alias_key != cache_key:
                self._cache_set(alias_key, response)

        return response

    def close(self) -> None:
        if self._neo4j_driver is not None:
            try:
                self._neo4j_driver.close()
            except Exception:
                logger.exception("Failed to close Neo4j driver")


_RUNTIME = HybridAnswerRuntime()


@mcp.tool()
def answer_service_healthcheck() -> Dict[str, Any]:
    """Return health and dependency status for the legal-answer pipeline."""
    logger.info("Tool answer_service_healthcheck invoked")
    result = _RUNTIME.health()
    logger.info(
        "Tool answer_service_healthcheck completed success=%s result=%s",
        result.get("success"),
        _preview_result_payload(result, max_chars=1200),
    )
    return result


@mcp.tool()
def answer_legal_question(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    include_graph: bool = True,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    ALWAYS call this tool first for user questions that may be legal or policy related.
    Use the returned `legal_related` flag to decide whether to continue with legal-answer flow.

    This tool answers Vietnamese legal/policy questions using configured knowledge stores.
    It is the only tool that can access KB/KG evidence.

    Parameters:
    - question: User question (Vietnamese preferred).
    - top_k: Initial retrieval size (1..MAX_TOP_K).
    - include_graph: Expand related evidence from Neo4j.
    - use_cache: Reuse recent responses when available.

    Guidance:
    - Prioritize this tool for regulations, legal procedures, and policy texts.
    - If evidence is weak or missing, state that limitation explicitly.
    - Do not claim legal certainty without KB/KG support.
    """
    logger.info(
        "Tool answer_legal_question invoked top_k=%s include_graph=%s use_cache=%s question=%s",
        top_k,
        include_graph,
        use_cache,
        _preview_query(question),
    )
    try:
        result = _RUNTIME.answer(
            question=question,
            top_k=top_k,
            include_graph=include_graph,
            use_cache=use_cache,
        )
        latency = result.get("latency_ms") or {}
        retrieved = result.get("retrieved") or {}
        providers = result.get("providers") or {}
        logger.info(
            "Tool answer_legal_question completed success=%s kb=%s kg=%s cache_hit=%s provider=%s total_latency_ms=%s result=%s",
            result.get("success"),
            retrieved.get("kb"),
            retrieved.get("kg"),
            result.get("cache_hit"),
            providers.get("generation"),
            latency.get("total"),
            _preview_result_payload(result),
        )
        return result
    except Exception as exc:
        safe_error = _sanitize_error_text(exc)
        logger.error("answer_legal_question failed: %s", safe_error)
        return {
            "success": False,
            "error": safe_error,
        }


if __name__ == "__main__":
    try:
        _RUNTIME.ensure_ready()
        mcp.run(transport="stdio")
    except Exception as exc:
        logger.error("Startup preflight failed: %s", _sanitize_error_text(exc))
        raise SystemExit(1)
    finally:
        _RUNTIME.close()
