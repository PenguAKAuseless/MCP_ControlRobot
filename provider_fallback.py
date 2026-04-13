from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import httpx

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


REMOTE_PROVIDER_PRIORITY = ["openai", "claude", "gemini", "togetherai"]
LOCAL_PROVIDER = "local"
DEFAULT_LOCAL_LLM_MODEL = "llama3.1:8b"
DEFAULT_LOCAL_EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_CAPABLE_PROVIDERS = {"openai", "gemini", "togetherai"}
PROVIDER_ALIASES = {
    "openai": "openai",
    "claude": "claude",
    "anthropic": "claude",
    "gemini": "gemini",
    "google": "gemini",
    "together": "togetherai",
    "togetherai": "togetherai",
    "local": "local",
    "ollama": "local",
}


class ProviderSetupError(RuntimeError):
    pass


def _first_non_empty(*values: Optional[str]) -> str:
    for value in values:
        if value and value.strip():
            return value.strip()
    return ""


def _env_flag(name: str, default: str = "0") -> bool:
    value = str(os.getenv(name, default)).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _normalize_provider_name(value: Optional[str]) -> str:
    raw = str(value or "").strip().lower()
    return PROVIDER_ALIASES.get(raw, "")


def _canonical_or_alias(canonical: str, *aliases: str) -> str:
    # If canonical key exists (even empty), treat it as authoritative.
    if canonical in os.environ:
        return str(os.getenv(canonical, "")).strip()
    return _first_non_empty(*(os.getenv(alias) for alias in aliases))


def _response_error_text(response: httpx.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            for key in ("error", "message", "detail"):
                value = payload.get(key)
                if value:
                    return str(value)
    except Exception:
        pass

    try:
        return str(response.text or "")
    except Exception:
        return ""


def _is_model_not_found_404(response: httpx.Response) -> bool:
    if response.status_code != 404:
        return False
    error_text = _response_error_text(response).lower()
    return "model" in error_text and "not found" in error_text


def _is_endpoint_not_found_404(response: httpx.Response) -> bool:
    if response.status_code != 404:
        return False

    error_text = _response_error_text(response).lower()
    if not error_text:
        return True

    if "model" in error_text and "not found" in error_text:
        return False

    return "not found" in error_text


class ProviderClientFallback:
    def __init__(
        self,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        mode_env = str(os.getenv("MCP_MODE", "")).strip().lower()
        self.local_mode = _env_flag("MCP_LOCAL_MODE", "0") or mode_env == "local"

        timeout_override = str(os.getenv("MCP_PROVIDER_TIMEOUT_SECONDS", "")).strip()
        if timeout_override:
            try:
                self.timeout_seconds = max(1.0, float(timeout_override))
            except ValueError:
                self.timeout_seconds = timeout_seconds
        else:
            self.timeout_seconds = max(timeout_seconds, 180.0) if self.local_mode else timeout_seconds

        self.preferred_generation_provider = _normalize_provider_name(
            _first_non_empty(
                os.getenv("MCP_PREFERRED_GENERATION_PROVIDER"),
                os.getenv("MCP_PREFERRED_PROVIDER"),
            )
        )
        self.preferred_embedding_provider = _normalize_provider_name(
            _first_non_empty(
                os.getenv("MCP_PREFERRED_EMBEDDING_PROVIDER"),
                os.getenv("MCP_PREFERRED_PROVIDER"),
            )
        )
        self.strict_preferred_provider = _env_flag("MCP_STRICT_PREFERRED_PROVIDER", "0")

        global_llm_if_set = str(os.getenv("MCP_LLM_MODEL", "")).strip() if "MCP_LLM_MODEL" in os.environ else ""
        global_embedding_if_set = str(os.getenv("MCP_EMBEDDING_MODEL", "")).strip() if "MCP_EMBEDDING_MODEL" in os.environ else ""

        self.local_base_url = _first_non_empty(
            os.getenv("MCP_LOCAL_BASE_URL"),
            os.getenv("MCP_OLLAMA_BASE_URL"),
            "http://127.0.0.1:11434",
        ).rstrip("/")
        self.local_llm_model = _first_non_empty(
            os.getenv("MCP_LOCAL_LLM_MODEL"),
            global_llm_if_set,
            DEFAULT_LOCAL_LLM_MODEL,
        )
        self.local_embedding_model = _first_non_empty(
            os.getenv("MCP_LOCAL_EMBEDDING_MODEL"),
            global_embedding_if_set,
            DEFAULT_LOCAL_EMBEDDING_MODEL,
        )

        if self.local_mode:
            self.preferred_generation_provider = LOCAL_PROVIDER
            self.preferred_embedding_provider = LOCAL_PROVIDER
            self.strict_preferred_provider = True

        self.keys: Dict[str, str] = {
            "openai": _canonical_or_alias(
                "MCP_OPENAI_API_KEY",
                "OPENAI_API_KEY",
            ),
            "claude": _canonical_or_alias(
                "MCP_CLAUDE_API_KEY",
                "MCP_ANTHROPIC_API_KEY",
                "ANTHROPIC_API_KEY",
            ),
            "gemini": _canonical_or_alias(
                "MCP_GEMINI_API_KEY",
                "GOOGLE_API_KEY",
                "GEMINI_API_KEY",
            ),
            "togetherai": _canonical_or_alias(
                "MCP_TOGETHER_API_KEY",
                "TOGETHER_API_KEY",
            ),
        }

        self.llm_models: Dict[str, str] = {
            "openai": _first_non_empty(
                os.getenv("MCP_OPENAI_LLM_MODEL"),
                llm_model,
                os.getenv("MCP_LLM_MODEL"),
                "gpt-4o-mini",
            ),
            "claude": _first_non_empty(
                os.getenv("MCP_CLAUDE_LLM_MODEL"),
                "claude-3-5-sonnet-latest",
            ),
            "gemini": _first_non_empty(
                os.getenv("MCP_GEMINI_LLM_MODEL"),
                "gemini-2.0-flash",
            ),
            "togetherai": _first_non_empty(
                os.getenv("MCP_TOGETHER_LLM_MODEL"),
                "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            ),
        }

        self.embedding_models: Dict[str, str] = {
            "openai": _first_non_empty(
                os.getenv("MCP_OPENAI_EMBEDDING_MODEL"),
                embedding_model,
                os.getenv("MCP_EMBEDDING_MODEL"),
                "text-embedding-3-small",
            ),
            "gemini": _first_non_empty(
                os.getenv("MCP_GEMINI_EMBEDDING_MODEL"),
                "gemini-embedding-001",
            ),
            "togetherai": _first_non_empty(
                os.getenv("MCP_TOGETHER_EMBEDDING_MODEL"),
                "togethercomputer/m2-bert-80M-32k-retrieval",
            ),
        }

        self.base_urls: Dict[str, str] = {
            "togetherai": _first_non_empty(
                os.getenv("MCP_TOGETHER_BASE_URL"),
                "https://api.together.xyz/v1",
            ),
        }
        self.gemini_base_url = _first_non_empty(
            os.getenv("MCP_GEMINI_BASE_URL"),
            "https://generativelanguage.googleapis.com/v1beta",
        )

        self._openai_clients: Dict[str, Any] = {}
        self.last_generation_provider: Optional[str] = None
        self.last_embedding_provider: Optional[str] = None

    def available_generation_providers(self) -> List[str]:
        if self.local_mode:
            return [LOCAL_PROVIDER]
        return [provider for provider in REMOTE_PROVIDER_PRIORITY if self.keys.get(provider)]

    def available_embedding_providers(self) -> List[str]:
        if self.local_mode:
            return [LOCAL_PROVIDER]
        return [
            provider
            for provider in REMOTE_PROVIDER_PRIORITY
            if provider in EMBEDDING_CAPABLE_PROVIDERS and self.keys.get(provider)
        ]

    def status(self) -> Dict[str, Any]:
        return {
            "mode": "local" if self.local_mode else "remote",
            "priority": [LOCAL_PROVIDER] if self.local_mode else list(REMOTE_PROVIDER_PRIORITY),
            "available_generation_providers": self.available_generation_providers(),
            "available_embedding_providers": self.available_embedding_providers(),
            "active_generation_provider": self.last_generation_provider,
            "active_embedding_provider": self.last_embedding_provider,
            "preferred_generation_provider": self.preferred_generation_provider,
            "preferred_embedding_provider": self.preferred_embedding_provider,
            "strict_preferred_provider": self.strict_preferred_provider,
            "local": {
                "enabled": self.local_mode,
                "base_url": self.local_base_url,
                "llm_model": self.local_llm_model,
                "embedding_model": self.local_embedding_model,
            },
            "models": {
                "llm": dict(self.llm_models),
                "embedding": dict(self.embedding_models),
            },
        }

    def validate(self, require_generation: bool, require_embeddings: bool) -> None:
        if self.local_mode:
            if require_generation and not self.local_llm_model:
                raise ProviderSetupError("Local mode requires MCP_LOCAL_LLM_MODEL")
            if require_embeddings and not self.local_embedding_model:
                raise ProviderSetupError("Local mode requires MCP_LOCAL_EMBEDDING_MODEL")

        if require_generation and not self.available_generation_providers():
            if self.local_mode:
                raise ProviderSetupError(
                    "Local mode is enabled but local model is unavailable. "
                    "Check MCP_LOCAL_LLM_MODEL and ensure Ollama is running."
                )
            raise ProviderSetupError(
                "No generation provider key found. Configure one of: "
                "MCP_OPENAI_API_KEY, MCP_CLAUDE_API_KEY, MCP_GEMINI_API_KEY, MCP_TOGETHER_API_KEY"
            )

        if require_generation and self.strict_preferred_provider and self.preferred_generation_provider:
            if self.preferred_generation_provider not in self.available_generation_providers():
                raise ProviderSetupError(
                    f"Preferred generation provider '{self.preferred_generation_provider}' is not available"
                )

        if require_embeddings and not self.available_embedding_providers():
            if self.local_mode:
                raise ProviderSetupError(
                    "Local mode is enabled but local embedding model is unavailable. "
                    "Check MCP_LOCAL_EMBEDDING_MODEL and ensure Ollama is running."
                )
            raise ProviderSetupError(
                "No embedding-capable provider key found. Configure one of: "
                "MCP_OPENAI_API_KEY, MCP_GEMINI_API_KEY, MCP_TOGETHER_API_KEY"
            )

        if require_embeddings and self.strict_preferred_provider and self.preferred_embedding_provider:
            if self.preferred_embedding_provider not in self.available_embedding_providers():
                raise ProviderSetupError(
                    f"Preferred embedding provider '{self.preferred_embedding_provider}' is not available"
                )

    def _ordered_candidates(self, providers: List[str], preferred: Optional[str]) -> List[str]:
        if not preferred:
            return list(providers)
        if preferred not in providers:
            return list(providers)
        ordered = [preferred]
        for provider in providers:
            if provider != preferred:
                ordered.append(provider)
        return ordered

    def _candidate_order(
        self,
        providers: List[str],
        sticky_provider: Optional[str],
        preferred_provider: Optional[str],
    ) -> List[str]:
        sticky = _normalize_provider_name(sticky_provider)
        preferred = _normalize_provider_name(preferred_provider)

        if self.strict_preferred_provider and preferred:
            if preferred in providers:
                return [preferred]
            return []

        return self._ordered_candidates(providers, sticky or preferred)

    def _openai_client_for(self, provider: str) -> Any:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")

        cached = self._openai_clients.get(provider)
        if cached is not None:
            return cached

        key = self.keys.get(provider, "")
        if not key:
            raise RuntimeError(f"Missing key for provider: {provider}")

        kwargs: Dict[str, Any] = {"api_key": key}
        if provider in self.base_urls:
            kwargs["base_url"] = self.base_urls[provider]

        client = OpenAI(**kwargs)
        self._openai_clients[provider] = client
        return client

    def _anthropic_generate(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: Optional[int] = None,
    ) -> str:
        api_key = self.keys.get("claude", "")
        if not api_key:
            raise RuntimeError("MCP_CLAUDE_API_KEY is not configured")

        model = self.llm_models["claude"]
        configured_max_tokens = max(128, int(os.getenv("MCP_CLAUDE_MAX_TOKENS", "1200")))
        effective_max_tokens = configured_max_tokens if max_tokens is None else max(1, int(max_tokens))

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        parts = data.get("content") or []
        text_parts = [str(item.get("text") or "") for item in parts if item.get("type") == "text"]
        output = "".join(text_parts).strip()
        if not output:
            raise RuntimeError("Claude returned empty content")
        return output

    def _gemini_generate(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: Optional[int] = None,
    ) -> str:
        api_key = self.keys.get("gemini", "")
        if not api_key:
            raise RuntimeError("MCP_GEMINI_API_KEY is not configured")

        model = self.llm_models["gemini"]
        url = f"{self.gemini_base_url}/models/{model}:generateContent?key={api_key}"
        payload = {
            "system_instruction": {
                "parts": [{"text": system}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
            },
        }
        if max_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = max(1, int(max_tokens))

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        candidates = data.get("candidates") or []
        for candidate in candidates:
            content = candidate.get("content") or {}
            parts = content.get("parts") or []
            text = "".join(str(part.get("text") or "") for part in parts).strip()
            if text:
                return text

        raise RuntimeError("Gemini returned empty content")

    def _gemini_embed_texts(self, texts: List[str]) -> List[List[float]]:
        api_key = self.keys.get("gemini", "")
        if not api_key:
            raise RuntimeError("MCP_GEMINI_API_KEY is not configured")

        model = self.embedding_models["gemini"]
        url = f"{self.gemini_base_url}/models/{model}:embedContent?key={api_key}"

        vectors: List[List[float]] = []
        with httpx.Client(timeout=self.timeout_seconds) as client:
            for text in texts:
                payload = {
                    "content": {
                        "parts": [{"text": text}],
                    }
                }
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                values = ((data.get("embedding") or {}).get("values") or [])
                if not values:
                    raise RuntimeError("Gemini embedding response was empty")
                vectors.append([float(value) for value in values])

        return vectors

    def _local_generate(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: Optional[int] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.local_llm_model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max(1, int(max_tokens))

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(f"{self.local_base_url}/api/generate", json=payload)
            if _is_model_not_found_404(response):
                raise RuntimeError(
                    f"Local generation model '{self.local_llm_model}' is not installed in Ollama. "
                    f"Run: ollama pull {self.local_llm_model}"
                )
            response.raise_for_status()
            data = response.json()

        output = str(data.get("response") or "").strip()
        if not output:
            raise RuntimeError("Local model returned empty content")
        return output

    def _local_embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        with httpx.Client(timeout=self.timeout_seconds) as client:
            for text in texts:
                response = client.post(
                    f"{self.local_base_url}/api/embed",
                    json={
                        "model": self.local_embedding_model,
                        "input": text,
                    },
                )
                if _is_model_not_found_404(response):
                    raise RuntimeError(
                        f"Local embedding model '{self.local_embedding_model}' is not installed in Ollama. "
                        f"Run: ollama pull {self.local_embedding_model}"
                    )

                if _is_endpoint_not_found_404(response):
                    response = client.post(
                        f"{self.local_base_url}/api/embeddings",
                        json={
                            "model": self.local_embedding_model,
                            "prompt": text,
                        },
                    )

                if _is_model_not_found_404(response):
                    raise RuntimeError(
                        f"Local embedding model '{self.local_embedding_model}' is not installed in Ollama. "
                        f"Run: ollama pull {self.local_embedding_model}"
                    )

                response.raise_for_status()
                data = response.json()

                vector: Any = None
                embeddings = data.get("embeddings")
                if isinstance(embeddings, list) and embeddings:
                    first = embeddings[0]
                    if isinstance(first, list):
                        vector = first
                    elif isinstance(first, dict):
                        vector = first.get("embedding")

                if vector is None:
                    vector = data.get("embedding")

                if not vector:
                    raise RuntimeError("Local embedding response was empty")

                vectors.append([float(value) for value in vector])

        return vectors

    def generate_text(
        self,
        prompt: str,
        system: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, str]:
        providers = self.available_generation_providers()
        if not providers:
            raise ProviderSetupError("No available generation providers")

        candidates = self._candidate_order(
            providers,
            self.last_generation_provider,
            self.preferred_generation_provider,
        )
        if not candidates:
            raise ProviderSetupError(
                f"Preferred generation provider '{self.preferred_generation_provider}' is not available"
            )

        errors: List[str] = []
        for provider in candidates:
            try:
                if provider == LOCAL_PROVIDER:
                    output = self._local_generate(
                        prompt=prompt,
                        system=system,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                elif provider == "claude":
                    output = self._anthropic_generate(
                        prompt=prompt,
                        system=system,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                elif provider == "gemini":
                    output = self._gemini_generate(
                        prompt=prompt,
                        system=system,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                else:
                    client = self._openai_client_for(provider)
                    request_kwargs: Dict[str, Any] = {
                        "model": self.llm_models[provider],
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": temperature,
                    }
                    if max_tokens is not None:
                        request_kwargs["max_tokens"] = max(1, int(max_tokens))
                    response = client.chat.completions.create(**request_kwargs)
                    output = str(response.choices[0].message.content or "").strip()
                    if not output:
                        raise RuntimeError("Model returned empty content")

                self.last_generation_provider = provider
                return output, provider
            except Exception as exc:
                errors.append(f"{provider}: {exc}")

        raise RuntimeError("All generation providers failed: " + " | ".join(errors))

    def embed_texts(self, texts: List[str], dimensions: Optional[int] = None) -> Tuple[List[List[float]], str]:
        providers = self.available_embedding_providers()
        if not providers:
            raise ProviderSetupError("No available embedding providers")

        candidates = self._candidate_order(
            providers,
            self.last_embedding_provider,
            self.preferred_embedding_provider,
        )
        if not candidates:
            raise ProviderSetupError(
                f"Preferred embedding provider '{self.preferred_embedding_provider}' is not available"
            )

        errors: List[str] = []
        for provider in candidates:
            try:
                if provider == LOCAL_PROVIDER:
                    vectors = self._local_embed_texts(texts)
                elif provider == "gemini":
                    vectors = self._gemini_embed_texts(texts)
                else:
                    client = self._openai_client_for(provider)
                    kwargs: Dict[str, Any] = {
                        "model": self.embedding_models[provider],
                        "input": texts,
                    }
                    if dimensions is not None and provider == "openai":
                        kwargs["dimensions"] = dimensions

                    response = client.embeddings.create(**kwargs)
                    vectors = [list(item.embedding) for item in response.data]
                if not vectors:
                    raise RuntimeError("Embedding response was empty")

                self.last_embedding_provider = provider
                return vectors, provider
            except Exception as exc:
                errors.append(f"{provider}: {exc}")

        raise RuntimeError("All embedding providers failed: " + " | ".join(errors))

    def embed_query(self, text: str, dimensions: Optional[int] = None) -> Tuple[List[float], str]:
        vectors, provider = self.embed_texts([text], dimensions=dimensions)
        return vectors[0], provider

    def verify_connectivity(self, verify_generation: bool = True, verify_embeddings: bool = True) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "generation_provider": None,
            "embedding_provider": None,
            "embedding_dimensions": None,
        }

        if verify_generation:
            _, generation_provider = self.generate_text(
                prompt="Respond with exactly: OK",
                system="Healthcheck. Respond with exactly: OK",
                temperature=0.0,
                max_tokens=8,
            )
            result["generation_provider"] = generation_provider

        if verify_embeddings:
            vector, embedding_provider = self.embed_query("healthcheck")
            result["embedding_provider"] = embedding_provider
            result["embedding_dimensions"] = len(vector)

        return result
