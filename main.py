from typedefinitions import *
from anthropic_inference_source import AnthropicExecutableTranslator
from openrouter_inference_source import (
    OpenrouterExecutableTranslator,
    OpenrouterGenericInference,
)
from groq_inference_source import GroqExecutableTranslator
from deepl_inference_source import DeeplExecutableTranslator
from lingvanex_inference_source import LingvanexExecutableTranslator
from mistral_inference_source import MistralExecutableTranslator
from nuenki_inference_source import NuenkiHybridExecutableTranslator

from secrets_env import *

target_languages = [lang for lang in TranslatableLanguage if lang.value != "English"]


# TODO ADD MAVERICK; WELSH; THAI.

# define list of models we're evaluating
evaluation_targets = [
    # 1 of 21
    TestedEntry(
        model_name=ModelName.Claude_Sonnet_35_2024_10_22,
        model_company=ModelCompany.Anthropic,
        inference_service_name=InferenceCompany.Anthropic,
        inference_source=AnthropicExecutableTranslator(
            ANTHROPIC_API_KEY,
            "claude-3-5-sonnet-20241022",
        ),
    ),
    # 2 of 21
    TestedEntry(
        model_name=ModelName.Claude_Sonnet_37_2025_02_19,
        model_company=ModelCompany.Anthropic,
        inference_service_name=InferenceCompany.Anthropic,
        inference_source=AnthropicExecutableTranslator(
            ANTHROPIC_API_KEY,
            "claude-3-7-sonnet-20250219",
        ),
    ),
    # 3 of 21
    TestedEntry(
        model_name=ModelName.DeepL,
        model_company=ModelCompany.DeepL,
        inference_service_name=InferenceCompany.DeepL,
        inference_source=DeeplExecutableTranslator(DEEPL_API_KEY),
    ),
    # 4 of 21
    TestedEntry(
        model_name=ModelName.Gemini_25_Flash_Preview_04_17,
        model_company=ModelCompany.Google,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "gemini-2-5-flash-preview-04-17",
        ),
    ),
    # 5 of 21
    TestedEntry(
        model_name=ModelName.Gemma3_27B,
        model_company=ModelCompany.Google,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "gemma-3-27b",
        ),
    ),
    # 6 of 21
    TestedEntry(
        model_name=ModelName.Grok_3_Beta,
        model_company=ModelCompany.X_AI,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "grok-3-beta",
        ),
    ),
    # 7 of 21
    TestedEntry(
        model_name=ModelName.GPT_41,
        model_company=ModelCompany.OpenAI,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "gpt-4.1",
        ),
    ),
    # 8 of 21
    TestedEntry(
        model_name=ModelName.GPT_41_Mini,
        model_company=ModelCompany.OpenAI,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "gpt-4.1-mini",
        ),
    ),
    # 9 of 21
    TestedEntry(
        model_name=ModelName.GPT_41_Nano,
        model_company=ModelCompany.OpenAI,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "gpt-4.1-nano",
        ),
    ),
    # 10 of 21
    TestedEntry(
        model_name=ModelName.GPT_4o,
        model_company=ModelCompany.OpenAI,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "gpt-4o",
        ),
    ),
    # 11 of 21
    TestedEntry(
        model_name=ModelName.Llama33_70b,
        model_company=ModelCompany.Meta,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "llama-3.3-70b",
        ),
    ),
    # 12 of 21
    TestedEntry(
        model_name=ModelName.Llama_31_8b,
        model_company=ModelCompany.Meta,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "llama-3.1-8b-instant",
        ),
    ),
    # 13 of 21
    TestedEntry(
        model_name=ModelName.Llama_4_Scout,
        model_company=ModelCompany.Meta,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "llama-4-scout",
        ),
    ),
    # 14 of 21
    TestedEntry(
        model_name=ModelName.Lingvanex,
        model_company=ModelCompany.Lingvanex,
        inference_service_name=InferenceCompany.Lingvanex,
        inference_source=LingvanexExecutableTranslator(LINGVANEX_API_KEY),
    ),
    # 15 of 21
    TestedEntry(
        model_name=ModelName.Mistral_Saba_24B,
        model_company=ModelCompany.Mistral,
        inference_service_name=InferenceCompany.Mistral,
        inference_source=MistralExecutableTranslator(
            MISTRAL_API_KEY,
            "mistral-saba-24b",
        ),
    ),
    # 16 of 21
    TestedEntry(
        model_name=ModelName.Mistral_Small_Latest,
        model_company=ModelCompany.Mistral,
        inference_service_name=InferenceCompany.Mistral,
        inference_source=MistralExecutableTranslator(
            MISTRAL_API_KEY,
            "mistral-small-latest",
        ),
    ),
    # 17 of 21
    TestedEntry(
        model_name=ModelName.Qwen_25_32B,
        model_company=ModelCompany.Alibaba,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "qwen-2-5-32b",
        ),
    ),
    # 18 of 21
    TestedEntry(
        model_name=ModelName.Qwen3_14b,
        model_company=ModelCompany.Alibaba,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "qwen-3-14b",
        ),
    ),
    # 19 of 21
    TestedEntry(
        model_name=ModelName.Qwen3_30_a3b,
        model_company=ModelCompany.Alibaba,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "qwen-3-30-a3b",
        ),
    ),
    # 20 of 21
    TestedEntry(
        model_name=ModelName.Qwen3_32b,
        model_company=ModelCompany.Alibaba,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "qwen-3-32b",
        ),
    ),
    # 21 of 21
    TestedEntry(
        model_name=ModelName.Qwen3_235b_a22b,
        model_company=ModelCompany.Alibaba,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "qwen-3-235b-a22b",
        ),
    ),
]


# define scoring methods
scoring_models = [
    OpenrouterGenericInference(OPENROUTER_API_KEY, "openai/gpt-4.1"),
    OpenrouterGenericInference(OPENROUTER_API_KEY, "openai/gpt-4o-2024-11-20"),
    OpenrouterGenericInference(OPENROUTER_API_KEY, "x-ai/grok-3-beta"),
]

# generate permutations to compare

# run inference and scoring. Obviously heavily heavily cache!

# produce option permutations and calculate scores
