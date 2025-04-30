from typedefinitions import *
from groq import Groq
from utils import generate_translation_prompt


class GroqExecutableTranslator(AbstractExecutableTranslator):
    def __init__(self, api_key: str, model_slug: str):
        self.client = Groq(api_key=api_key)
        self.model_slug = model_slug

    def is_tl_valid(self, target_lang: TranslatableLanguage) -> bool:
        return True

    def translate(
        self,
        source_lang: TranslatableLanguage,
        target_lang: TranslatableLanguage,
        text: str,
        temperature: float,
    ) -> str:
        prompt = generate_translation_prompt(source_lang, target_lang, text)

        response = self.client.chat.completions.create(
            model=self.model_slug,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content
