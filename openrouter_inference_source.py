from typedefinitions import *
import requests
from utils import generate_translation_prompt

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenrouterExecutableTranslator(AbstractExecutableTranslator):
    def __init__(self, api_key: str, model_slug: str):
        self.model_slug = model_slug
        self.api_key = api_key

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

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_slug,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"OpenRouter error {response.status_code}: {response.text}")

        result = response.json()
        return result["choices"][0]["message"]["content"]


class OpenrouterGenericInference(AbstractGenericInference):
    def __init__(self, api_key: str, model_slug: str):
        self.api_key = api_key
        self.model_slug = model_slug

    def infer(
        self,
        user_prompt: str,
        temperature: float,
    ) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_slug,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        }

        response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"OpenRouter error {response.status_code}: {response.text}")

        data = response.json()
        return data["choices"][0]["message"]["content"]
