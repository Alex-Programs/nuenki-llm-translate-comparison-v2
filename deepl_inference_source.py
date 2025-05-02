from typedefinitions import *
import requests


class DeeplExecutableTranslator(AbstractExecutableTranslator):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._url = "https://api.deepl.com/v2/translate"

    def is_tl_valid(self, target_lang: TranslatableLanguage) -> bool:
        try:
            _ = target_lang.deepl_code()
            return True
        except KeyError:
            return False

    def translate(
        self,
        source_lang: TranslatableLanguage,
        target_lang: TranslatableLanguage,
        text: str,
        temperature: float,
    ) -> str:
        try:
            source_code = source_lang.deepl_code()
            target_code = target_lang.deepl_code()
        except KeyError as e:
            raise ValueError(f"Unsupported language: {e.args[0]}")

        headers = {"Authorization": f"DeepL-Auth-Key {self.api_key}"}
        data = {
            "text": text,
            "source_lang": source_code,
            "target_lang": target_code,
        }

        response = requests.post(self._url, data=data, headers=headers)
        if response.status_code != 200:
            raise Exception(f"DeepL error {response.status_code}: {response.text}")

        payload = response.json()
        translations = payload.get("translations")
        if not translations:
            raise Exception(f"DeepL returned unexpected payload: {payload}")

        return translations[0]["text"]
