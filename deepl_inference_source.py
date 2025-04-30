from typedefinitions import *
import requests


class DeeplExecutableTranslator(AbstractExecutableTranslator):
    _LANG_MAP = {
        TranslatableLanguage.English: "EN",
        TranslatableLanguage.German: "DE",
        TranslatableLanguage.Chinese: "ZH",
        TranslatableLanguage.Hungarian: "HU",
        TranslatableLanguage.French: "FR",
        TranslatableLanguage.Japanese: "JA",
        TranslatableLanguage.Italian: "IT",
        TranslatableLanguage.EuropeanSpanish: "ES",
        TranslatableLanguage.Ukrainian: "UK",
        TranslatableLanguage.Swedish: "SV",
        TranslatableLanguage.Korean: "KO",
        TranslatableLanguage.Dutch: "NL",
        TranslatableLanguage.Polish: "PL",
        TranslatableLanguage.Portuguese: "PT",
        TranslatableLanguage.Russian: "RU",
    }

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._url = "https://api.deepl.com/v2/translate"

    def is_tl_valid(self, target_lang: TranslatableLanguage) -> bool:
        return target_lang in self._LANG_MAP

    def translate(
        self,
        source_lang: TranslatableLanguage,
        target_lang: TranslatableLanguage,
        text: str,
        temperature: float,
    ) -> str:
        if source_lang not in self._LANG_MAP or target_lang not in self._LANG_MAP:
            raise ValueError(f"Unsupported language: {source_lang} → {target_lang}")

        headers = {"Authorization": f"DeepL-Auth-Key {self.api_key}"}
        data = {
            "text": text,
            "source_lang": self._LANG_MAP[source_lang],
            "target_lang": self._LANG_MAP[target_lang],
        }

        response = requests.post(self._url, data=data, headers=headers)
        if response.status_code != 200:
            raise Exception(f"DeepL error {response.status_code}: {response.text}")

        payload = response.json()
        translations = payload.get("translations")
        if not translations:
            raise Exception(f"DeepL returned unexpected payload: {payload}")

        return translations[0]["text"]
