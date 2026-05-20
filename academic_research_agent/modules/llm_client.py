"""LLM client — thin wrapper around OpenAI-compatible chat APIs.

Supports: OpenAI, DeepSeek, Ollama, and any other OpenAI-compatible provider.
"""

import os
from openai import OpenAI


# Default model when none is configured
_DEFAULT_MODEL = "deepseek-chat"
_DEFAULT_BASE_URL = "https://api.deepseek.com"


class LLMClient:
    """Minimal OpenAI-compatible chat client.

    Usage::

        client = LLMClient()                        # reads LLM_API_KEY from env
        client = LLMClient(api_key="sk-...", base_url="https://api.deepseek.com")
        reply = client.chat([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ])
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        api_key = (
            api_key
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "LLM API key not configured. Set LLM_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self._api_key = api_key
        self._base_url = base_url or _DEFAULT_BASE_URL
        self._model = model or _DEFAULT_MODEL
        self._client = OpenAI(api_key=api_key, base_url=self._base_url)

    @property
    def model(self) -> str:
        return self._model

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        """Send a chat completion request and return the reply text.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-format messages: [{"role": "...", "content": "..."}]
        temperature : float
            Sampling temperature (default 0.3 for factual reports).
        max_tokens : int
            Maximum tokens in the response.

        Returns
        -------
        str — the assistant reply text.

        Raises
        ------
        RuntimeError — on API failure.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            content = response.choices[0].message.content
            return content or ""
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}") from e


# ── Singleton / factory ──────────────────────────────────────────────────

_client_instance: LLMClient | None = None


def get_llm_client(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> LLMClient:
    """Get or create a singleton LLMClient.

    If a client already exists with the same parameters, returns it.
    Otherwise creates a new one.
    """
    global _client_instance
    # Always create fresh if any param is explicitly provided
    if api_key or base_url or model:
        _client_instance = LLMClient(
            api_key=api_key, base_url=base_url, model=model
        )
    elif _client_instance is None:
        _client_instance = LLMClient()
    return _client_instance


def try_configure_llm(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> tuple[LLMClient | None, str | None]:
    """Attempt to configure the LLM client, returning (client, error_message).

    Never raises — returns None client and a user-facing error string on failure.
    """
    try:
        client = get_llm_client(api_key=api_key, base_url=base_url, model=model)
        return client, None
    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, f"LLM 客户端初始化失败: {e}"
