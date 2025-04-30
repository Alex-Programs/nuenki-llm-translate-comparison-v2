from dataclasses import dataclass, asdict
import json
from enum import Enum
from abc import ABC, abstractmethod


class TranslatableLanguage(Enum):
    English = "English"
    German = "German"
    Chinese = "Chinese"
    Hungarian = "Hungarian"
    French = "French"
    Japanese = "Japanese"
    Esperanto = "Esperanto"
    Italian = "Italian"
    EuropeanSpanish = "EuropeanSpanish"
    Ukrainian = "Ukrainian"
    Swedish = "Swedish"
    Korean = "Korean"
    Vietnamese = "Vietnamese"


class EnumSupportedEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Enum):
            return o.value
        return super().default(o)


class ModelName(Enum):
    GPT_41 = "GPT 4.1"
    DeepL = "DeepL"
    GPT_4o = "GPT 4o"
    Grok_3_Beta = "Grok 3 Beta"
    Claude_Sonnet_35_2024_10_22 = "Claude Sonnet 3.5 2024-10-22"
    Claude_Sonnet_37_2025_02_19 = "Claude Sonnet 3.7 2025-02-19"
    GPT_41_Mini = "GPT 4.1 Mini"
    Gemini_25_Flash_Preview_04_17 = "Gemini 2.5 Flash Preview 04-17"
    Gemma3_27B = "Gemma 3 27b"
    Lingvanex = "Lingvanex"
    Llama_4_Scout = "LLama 4 Scout"
    Llama33_70b = "Llama 3.3 70b"
    GPT_41_Nano = "GPT 4.1 Nano"
    Qwen_25_32B = "Qwen 2.5 32B"
    Llama_31_8b = "LLama 3.1 8b instant"
    Mistral_Small_Latest = "Mistral Small Latest"
    Mistral_Saba_24B = "Mistral Saba 24B"
    Qwen3_30_a3b = "Qwen 3 30B A3B"
    Qwen3_14b = "Qwen 3 14B"
    Qwen3_32b = "Qwen 3 32B"
    Qwen3_235b_a22b = "Qwen 3 32B A22B"


class ModelCompany(Enum):
    OpenAI = "OpenAI"
    DeepL = "DeepL"
    X_AI = "X AI"
    Anthropic = "Anthropic"
    Google = "Google"
    Lingvanex = "Lingvanex"
    Meta = "Meta"
    Alibaba = "Alibaba"
    Mistral = "Mistral"


class InferenceCompany(Enum):
    Groq = "Groq"
    Openrouter = "Openrouter"
    Mistral = "Mistral"
    Anthropic = "Anthropic"
    DeepL = "DeepL"
    Lingvanex = "Lingvanex"


# I'm not usually one for inheritance, but it works so nicely here!
class AbstractExecutableTranslator(ABC):
    @abstractmethod
    def translate(
        self,
        source_lang: TranslatableLanguage,
        target_lang: TranslatableLanguage,
        text: str,
        temperature: float,
    ):
        """Translate"""
        raise NotImplementedError

    @abstractmethod
    def is_tl_valid(self, target_lang: TranslatableLanguage):
        pass


class AbstractGenericInference(ABC):
    @abstractmethod
    def infer(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> str:
        """Run a chat completion with a system and user prompt at the given temperature."""
        raise NotImplementedError


@dataclass
class TestedEntry:
    model_name: ModelName
    model_company: ModelCompany
    inference_service_name: InferenceCompany
    inference_source: AbstractExecutableTranslator

    def unique_id(self):
        (
            self.model_name.value
            + self.model_company.value
            + self.inference_service_name.value
            + self.inference_source.value
        )
