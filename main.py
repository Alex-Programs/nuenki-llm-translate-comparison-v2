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
from google_inference_source import GoogleGenericInference

from secrets_env import *

from sqlitekv import SQLiteKVCache
from test_data import *
from run_evaluation import evaluate_datasets

cache = SQLiteKVCache("./cache.db")

# define comparing methods
compare_models = [
    (
        "openai/gpt-4.1-comparison-system",
        OpenrouterGenericInference(OPENROUTER_API_KEY, "openai/gpt-4.1"),
    ),
    (
        "anthropic/claude-3.7-sonnet-comparison-system",
        OpenrouterGenericInference(OPENROUTER_API_KEY, "anthropic/claude-3.7-sonnet"),
    ),
    (
        "x-ai/grok-3-beta-comparison-system",
        OpenrouterGenericInference(OPENROUTER_API_KEY, "x-ai/grok-3-beta"),
    ),
    (
        "google/gemini-2.5-pro-preview-05-06-comparison-system-thinking",
        GoogleGenericInference(GEMINI_API_KEY, "gemini-2.5-pro-preview-05-06", True),
    ),
]

# first run on testing dataset
evaluate_datasets(
    target_languages_testing, evaluation_targets_testing, cache, compare_models
)

# then run on sensible_large on German for a broad idea

# then run on the temperature testing dataset

# then run on the low_resource_inform on the best existing models on all niche Nuenki langs
