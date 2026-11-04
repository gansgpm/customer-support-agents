"""Model-client factory for every provider AutoGen supports through the OpenAI-compatible API."""
from __future__ import annotations

import os

from autogen_core.models import ChatCompletionClient, ModelFamily

from engine.config import Settings

# Capabilities for models AutoGen does not know by name (Groq, Ollama, ...).
_GENERIC_INFO = {
    "vision": False,
    "function_calling": True,
    "json_output": True,
    "family": ModelFamily.UNKNOWN,
    "structured_output": False,
    "multiple_system_messages": False,
}


class ProviderConfigError(RuntimeError):
    pass


def _require(var: str, provider: str) -> str:
    value = os.getenv(var, "").strip()
    if not value:
        raise ProviderConfigError(f"{var} is missing in .env (needed for LLM_PROVIDER={provider}). "
                                  f"Add it, or set LLM_PROVIDER=offline to run the demo without a key.")
    return value


def build_model_client(settings: Settings) -> ChatCompletionClient:
    provider = settings.llm_provider
    model = settings.model_name
    temperature = settings.llm_temperature
    # transient API errors (rate limits, timeouts, 5xx) are retried with back-off by the OpenAI SDK
    resilience = {"max_retries": settings.llm_max_retries, "timeout": settings.llm_timeout}

    if provider == "offline":
        from engine.offline_client import OfflineChatCompletionClient

        return OfflineChatCompletionClient()

    from autogen_ext.models.openai import AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient

    if provider == "openai":
        return OpenAIChatCompletionClient(model=model, api_key=_require("OPENAI_API_KEY", provider),
                                          temperature=temperature, **resilience)
    if provider == "azure":
        return AzureOpenAIChatCompletionClient(
            model=model,
            azure_deployment=_require("AZURE_OPENAI_DEPLOYMENT", provider),
            azure_endpoint=_require("AZURE_OPENAI_ENDPOINT", provider),
            api_key=_require("AZURE_OPENAI_API_KEY", provider),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            temperature=temperature, **resilience,
        )
    if provider == "gemini":
        info = None if model in {"gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-pro"} else {
            **_GENERIC_INFO, "family": ModelFamily.GEMINI_2_5_FLASH}
        kwargs = {"model_info": info} if info else {}
        return OpenAIChatCompletionClient(model=model, api_key=_require("GEMINI_API_KEY", provider),
                                          temperature=temperature, include_name_in_message=False, **kwargs,
                                          **resilience)
    if provider == "groq":
        return OpenAIChatCompletionClient(model=model, api_key=_require("GROQ_API_KEY", provider),
                                          base_url="https://api.groq.com/openai/v1", model_info=_GENERIC_INFO,
                                          temperature=temperature, include_name_in_message=False, **resilience)
    if provider == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        return OpenAIChatCompletionClient(model=model, api_key="ollama", base_url=f"{base}/v1",
                                          model_info=_GENERIC_INFO, temperature=temperature,
                                          include_name_in_message=False, **resilience)
    raise ProviderConfigError(f"Unknown LLM_PROVIDER '{provider}'. Use openai, azure, gemini, groq, ollama or offline.")


def describe(settings: Settings) -> str:
    if settings.is_offline:
        return "Offline demo model (rule-based, no API key)"
    return f"{settings.llm_provider} / {settings.model_name}"
