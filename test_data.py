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

target_languages_full = [
    lang for lang in TranslatableLanguage if lang.value != "English"
]
target_languages_testing = [TranslatableLanguage.German]

evaluation_targets_sensible_large_temp0_nothink = [
    TestedEntry(
        model_name=ModelName.Nuenki_Hybrid,
        model_company=ModelCompany.Nuenki,
        inference_service_name=InferenceCompany.Nuenki,
        inference_source=NuenkiHybridExecutableTranslator(NUENKI_API_KEY),
        temp=None,
        thinking="N/A",
    ),
    TestedEntry(
        model_name=ModelName.GPT_41,
        model_company=ModelCompany.OpenAI,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "openai/gpt-4.1",
        ),
        temp=0,
        thinking="N/A",
    ),
    TestedEntry(
        model_name=ModelName.GPT_4o,
        model_company=ModelCompany.OpenAI,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "openai/gpt-4o-2024-11-20",
        ),
        temp=0,
        thinking="N/A",
    ),
    TestedEntry(
        model_name=ModelName.Grok_3_Beta,
        model_company=ModelCompany.X_AI,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "x-ai/grok-3-beta",
        ),
        temp=0,
        thinking="N/A",
    ),
    TestedEntry(
        model_name=ModelName.Claude_Sonnet_37_2025_02_19,
        model_company=ModelCompany.Anthropic,
        inference_service_name=InferenceCompany.Anthropic,
        inference_source=AnthropicExecutableTranslator(
            ANTHROPIC_API_KEY,
            "claude-3-7-sonnet-20250219",
        ),
        temp=0,
        thinking="Off",
    ),
    TestedEntry(
        model_name=ModelName.Claude_Sonnet_35_2024_10_22,
        model_company=ModelCompany.Anthropic,
        inference_service_name=InferenceCompany.Anthropic,
        inference_source=AnthropicExecutableTranslator(
            ANTHROPIC_API_KEY,
            "claude-3-5-sonnet-20241022",
        ),
        temp=0,
        thinking="N/A",
    ),
    TestedEntry(
        model_name=ModelName.GPT_41_Mini,
        model_company=ModelCompany.OpenAI,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "openai/gpt-4.1-mini",
        ),
        temp=0,
        thinking="N/A",
    ),
    TestedEntry(
        model_name=ModelName.Gemini_25_Flash_Preview_04_17,
        model_company=ModelCompany.Google,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "google/gemini-2.5-flash-preview",
        ),
        temp=0,
        thinking="Disabled",
    ),
    TestedEntry(
        model_name=ModelName.DeepL,
        model_company=ModelCompany.DeepL,
        inference_service_name=InferenceCompany.DeepL,
        inference_source=DeeplExecutableTranslator(DEEPL_API_KEY),
        temp=None,
        thinking="N/A",
    ),
    TestedEntry(
        model_name=ModelName.Gemma3_27B,
        model_company=ModelCompany.Google,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "google/gemma-3-27b-it",
        ),
        temp=0,
        thinking="N/A",
    ),
    TestedEntry(
        model_name=ModelName.Lingvanex,
        model_company=ModelCompany.Lingvanex,
        inference_service_name=InferenceCompany.Lingvanex,
        inference_source=LingvanexExecutableTranslator(LINGVANEX_API_KEY),
        temp=None,
        thinking="N/A",
    ),
    TestedEntry(
        model_name=ModelName.Qwen3_235b_a22b,
        model_company=ModelCompany.Alibaba,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY, "qwen/qwen3-235b-a22b", also_add="/no_think\n"
        ),
        temp=0,
        thinking="Off",
    ),
    TestedEntry(
        model_name=ModelName.Qwen3_32b,
        model_company=ModelCompany.Alibaba,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY, "qwen/qwen3-32b", also_add="/no_think\n"
        ),
        temp=0,
        thinking="Off",
    ),
    TestedEntry(
        model_name=ModelName.Qwen3_30_a3b,
        model_company=ModelCompany.Alibaba,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY, "qwen/qwen3-30b-a3b", also_add="/no_think\n"
        ),
        temp=0.0,
        thinking="Off",
    ),
    # TestedEntry(
    #    model_name=ModelName.Qwen3_14b,
    #    model_company=ModelCompany.Alibaba,
    #    inference_service_name=InferenceCompany.Openrouter,
    #    inference_source=OpenrouterExecutableTranslator(
    #        OPENROUTER_API_KEY, "qwen/qwen3-14b", also_add="/no_think\n"
    #    ),
    #    temp=0,
    #    thinking="Off",
    # ),
    TestedEntry(
        model_name=ModelName.Llama_4_Maverick,
        model_company=ModelCompany.Meta,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "meta-llama/llama-4-maverick:free",
        ),
        temp=0,
        thinking="N/A",
    ),
    TestedEntry(
        model_name=ModelName.Llama_4_Scout,
        model_company=ModelCompany.Meta,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "meta-llama/llama-4-scout",
        ),
        temp=0,
        thinking="N/A",
    ),
    TestedEntry(
        model_name=ModelName.Llama33_70b,
        model_company=ModelCompany.Meta,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "meta-llama/llama-3.3-70b-instruct",
        ),
        temp=0,
        thinking="N/A",
    ),
    TestedEntry(
        model_name=ModelName.GPT_41_Nano,
        model_company=ModelCompany.OpenAI,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "openai/gpt-4.1-nano",
        ),
        temp=0,
        thinking="N/A",
    ),
    # TestedEntry(
    #    model_name=ModelName.Llama_31_8b,
    #    model_company=ModelCompany.Meta,
    #    inference_service_name=InferenceCompany.Openrouter,
    #    inference_source=OpenrouterExecutableTranslator(
    #        OPENROUTER_API_KEY,
    #        "llama-3.1-8b-instant",
    #    ),
    # ),
    TestedEntry(
        model_name=ModelName.Mistral_Small_Latest,
        model_company=ModelCompany.Mistral,
        inference_service_name=InferenceCompany.Mistral,
        inference_source=MistralExecutableTranslator(
            MISTRAL_API_KEY,
            "mistral-small-latest",
        ),
        temp=0,
        thinking="N/A",
    ),
    # TestedEntry(
    #    model_name=ModelName.Mistral_Saba_24B,
    #    model_company=ModelCompany.Mistral,
    #    inference_service_name=InferenceCompany.Mistral,
    #    inference_source=MistralExecutableTranslator(
    #        MISTRAL_API_KEY,
    #        "mistral-saba-24b",
    #    ),
    # ),
]


evaluation_targets_testing = [
    TestedEntry(
        model_name=ModelName.Nuenki_Hybrid,
        model_company=ModelCompany.Nuenki,
        inference_service_name=InferenceCompany.Nuenki,
        inference_source=NuenkiHybridExecutableTranslator(NUENKI_API_KEY),
        temp=None,
        thinking="N/A",
    ),
    TestedEntry(
        model_name=ModelName.GPT_41,
        model_company=ModelCompany.OpenAI,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "openai/gpt-4.1",
        ),
        temp=0,
        thinking="N/A",
    ),
    TestedEntry(
        model_name=ModelName.GPT_4o,
        model_company=ModelCompany.OpenAI,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "openai/gpt-4o-2024-11-20",
        ),
        temp=0,
        thinking="N/A",
    ),
    TestedEntry(
        model_name=ModelName.Grok_3_Beta,
        model_company=ModelCompany.X_AI,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "x-ai/grok-3-beta",
        ),
        temp=0,
        thinking="N/A",
    ),
    TestedEntry(
        model_name=ModelName.Claude_Sonnet_37_2025_02_19,
        model_company=ModelCompany.Anthropic,
        inference_service_name=InferenceCompany.Anthropic,
        inference_source=AnthropicExecutableTranslator(
            ANTHROPIC_API_KEY,
            "claude-3-7-sonnet-20250219",
        ),
        temp=0,
        thinking="Off",
    ),
    TestedEntry(
        model_name=ModelName.Claude_Sonnet_35_2024_10_22,
        model_company=ModelCompany.Anthropic,
        inference_service_name=InferenceCompany.Anthropic,
        inference_source=AnthropicExecutableTranslator(
            ANTHROPIC_API_KEY,
            "claude-3-5-sonnet-20241022",
        ),
        temp=0,
        thinking="N/A",
    ),
]
