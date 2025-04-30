from typedefinitions import *
import requests

LINGVANEX_URL = "https://api-b2b.backenster.com/b1/api/v3/translate/"


class LingvanexExecutableTranslator(AbstractExecutableTranslator):
    def __init__(self, api_key: str):
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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        data = {
            "from": get_lingvanex_code(source_lang),
            "to": get_lingvanex_code(target_lang),
            "text": text,
            "platform": "dp",
        }

        response = requests.post(LINGVANEX_URL, headers=headers, data=data)
        if response.status_code != 200:
            raise Exception(f"Lingvanex error {response.status_code}: {response.text}")

        json = response.json()
        if json["err"]:
            raise Exception(f"Lingvanex returned error: {json['err']}")

        return json["result"]


def get_lingvanex_code(lang: TranslatableLanguage) -> str:
    match lang:
        case TranslatableLanguage.English:
            return "en_US"
        case TranslatableLanguage.German:
            return "de_DE"
        case TranslatableLanguage.Chinese:
            return "zh-Hans_CN"
        case TranslatableLanguage.Hungarian:
            return "hu_HU"
        case TranslatableLanguage.French:
            return "fr_FR"
        case TranslatableLanguage.Japanese:
            return "ja_JP"
        case TranslatableLanguage.Esperanto:
            return "eo_WORLD"
        case TranslatableLanguage.Italian:
            return "it_IT"
        case TranslatableLanguage.EuropeanSpanish:
            return "es_ES"
        case TranslatableLanguage.Ukrainian:
            return "uk_UA"
        case TranslatableLanguage.Swedish:
            return "sv_SE"
        case TranslatableLanguage.Korean:
            return "ko_KR"
        case TranslatableLanguage.Vietnamese:
            return "vi_VN"
        case _:
            raise Exception(f"No Lingvanex code for language: {lang}")
