from typedefinitions import *
import requests
from utils import generate_translation_prompt
from google import genai

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class GoogleExecutableTranslator(AbstractExecutableTranslator):
    def __init__(self, api_key: str, model_slug: str, thinking: bool):
        self.model_slug = model_slug
        self.api_key = api_key
        self.client = genai.client(api_key=api_key)
        self.thinking = thinking

    def is_tl_valid(self, target_lang: TranslatableLanguage):
        return True

    def translate(
        self,
        source_lang: TranslatableLanguage,
        target_lang: TranslatableLanguage,
        text: str,
        temperature: float,
    ) -> str:
        prompt = generate_translation_prompt(source_lang, target_lang, text)

        think_budget = None
        if not self.thinking:
            think_budget = 0

        response = self.client.models.generate_content(
            model=self.model_slug,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=temperature,
                thinking_config=genai.types.ThinkingConfig(
                    thinking_budget=think_budget
                ),
            ),
        )

        return response.text


class GoogleGenericInference(AbstractGenericInference):
    def __init__(self, api_key: str, model_slug: str, thinking: bool):
        self.api_key = api_key
        self.model_slug = model_slug
        self.client = genai.client(api_key=api_key)
        self.thinking = thinking

    def infer(
        self,
        user_prompt: str,
        temperature: float,
    ) -> str:
        think_budget = None
        if not self.thinking:
            think_budget = 0

        response = self.client.models.generate_content(
            model=self.model_slug,
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(
                temperature=temperature,
                thinking_config=genai.types.ThinkingConfig(
                    thinking_budget=think_budget
                ),
            ),
        )

        return response.text
