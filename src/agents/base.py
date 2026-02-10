import json
from importlib import import_module
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()
from jinja2 import Environment, FileSystemLoader
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

PATH_TO_TEMPLATES = Path(__file__).parent / "prompts"

MESSAGE_CLASSES = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}

# Supported providers and their LangChain classes
PROVIDER_CLASSES: dict[str, str] = {
    "openai": "langchain_openai.ChatOpenAI",
    "anthropic": "langchain_anthropic.ChatAnthropic",
    "ollama": "langchain_ollama.ChatOllama",
}


def get_model(provider: str, model: str, **kwargs: Any) -> BaseChatModel:
    """Create a LangChain chat model from provider and model name."""
    if provider not in PROVIDER_CLASSES:
        raise ValueError(
            f"Unknown provider: {provider}. Supported: {list(PROVIDER_CLASSES.keys())}"
        )

    module_path, class_name = PROVIDER_CLASSES[provider].rsplit(".", 1)
    module = import_module(module_path)
    model_class = getattr(module, class_name)

    return model_class(model=model, **kwargs)


def custom_tojson(value: Any) -> str:
    """Sanitize and JSON-encode a value for prompt insertion."""
    sanitized = re.sub(r"[^\x20-\x7E]", " ", value) if isinstance(value, str) else value
    return json.dumps(sanitized, ensure_ascii=False)


def load_prompt(prompt_name: str, **variables: Any) -> list:
    """Load a Jinja2 YAML template and render to LangChain messages."""
    env = Environment(loader=FileSystemLoader(PATH_TO_TEMPLATES))
    env.filters["custom_tojson"] = custom_tojson
    env.globals["custom_tojson"] = custom_tojson
    template = env.get_template(f"{prompt_name}.yml.j2")
    rendered = template.render(**variables)
    messages_data = yaml.safe_load(rendered)
    return [MESSAGE_CLASSES[m["role"]](content=m["content"]) for m in messages_data]


class BaseAgent:
    """Base agent class with swappable LLM providers."""
    output_schema = None  # Subclasses must define this

    def __init__(
        self,
        provider: str,
        model: str,
        prompt_name: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self.provider = provider
        self.model_name = model
        self.prompt_name = prompt_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = get_model(
            provider,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._structured_llm = self.llm.with_structured_output(self.output_schema)

    def run_single(self, **variables: Any):
        """Run the agent with the given input variables."""
        messages = load_prompt(self.prompt_name, **variables)
        return self._structured_llm.invoke(messages)

    def run_batch(self, inputs: list[dict[str, Any]], max_concurrency: int = 4):
        """Run the agent on multiple inputs in parallel."""
        all_messages = [load_prompt(self.prompt_name, **inp) for inp in inputs]
        return self._structured_llm.batch(
            all_messages,
            config={"max_concurrency": max_concurrency},
        )
