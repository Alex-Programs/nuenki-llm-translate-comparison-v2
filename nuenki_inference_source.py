import requests
from typedefinitions import *


class NuenkiHybridExecutableTranslator(AbstractExecutableTranslator):
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.api_url = "https://api.nuenki.app/hybrid_translate"

    def is_tl_valid(self, target_lang: TranslatableLanguage):
        return True

    def translate(
        self,
        source_lang: TranslatableLanguage,
        target_lang: TranslatableLanguage,
        text: str,
        temperature: float,
    ) -> str:
        response = requests.post(
            self.api_url,
            json={
                "source_lang": source_lang,
                "target_lang": target_lang,
                "text": text,
                "formality": "NormalFormality",
                "token": self.api_token,
            },
        )

        result = response.json()
        return [x for x in result["translations"] if x["combined"]][0]["text"]
