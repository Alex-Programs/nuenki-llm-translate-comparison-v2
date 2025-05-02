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
from dataset import SENTENCES_LIST
from secrets_env import *
import hashlib
from sqlitekv import SQLiteKVCache
import time
import choix

from latency_logger import log_latency

target_languages = [
    TranslatableLanguage.German,
    # TranslatableLanguage.Chinese,
    # TranslatableLanguage.French,
    # TranslatableLanguage.Ukrainian,
    # TranslatableLanguage.Hungarian,
]  # [lang for lang in TranslatableLanguage if lang.value != "English"]

cache = SQLiteKVCache("./cache.db")

# define list of models we're evaluating
evaluation_targets = [
    TestedEntry(
        model_name=ModelName.GPT_41,
        model_company=ModelCompany.OpenAI,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "openai/gpt-4.1",
        ),
        temp=0.5,
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
    ),
    TestedEntry(
        model_name=ModelName.GPT_4o,
        model_company=ModelCompany.OpenAI,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "openai/gpt-4o-2024-11-20",
        ),
        temp=0.5,
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
    ),
    TestedEntry(
        model_name=ModelName.Grok_3_Beta,
        model_company=ModelCompany.X_AI,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "x-ai/grok-3-beta",
        ),
        temp=0.5,
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
    ),
    TestedEntry(
        model_name=ModelName.Claude_Sonnet_37_2025_02_19,
        model_company=ModelCompany.Anthropic,
        inference_service_name=InferenceCompany.Anthropic,
        inference_source=AnthropicExecutableTranslator(
            ANTHROPIC_API_KEY,
            "claude-3-7-sonnet-20250219",
        ),
        temp=0.5,
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
    ),
    TestedEntry(
        model_name=ModelName.Claude_Sonnet_35_2024_10_22,
        model_company=ModelCompany.Anthropic,
        inference_service_name=InferenceCompany.Anthropic,
        inference_source=AnthropicExecutableTranslator(
            ANTHROPIC_API_KEY,
            "claude-3-5-sonnet-20241022",
        ),
        temp=0.5,
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
    ),
    TestedEntry(
        model_name=ModelName.GPT_41_Mini,
        model_company=ModelCompany.OpenAI,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "openai/gpt-4.1-mini",
        ),
        temp=0.5,
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
    ),
    TestedEntry(
        model_name=ModelName.Gemini_25_Flash_Preview_04_17,
        model_company=ModelCompany.Google,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "google/gemini-2.5-flash-preview",
        ),
        temp=0.5,
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
    ),
    TestedEntry(
        model_name=ModelName.DeepL,
        model_company=ModelCompany.DeepL,
        inference_service_name=InferenceCompany.DeepL,
        inference_source=DeeplExecutableTranslator(DEEPL_API_KEY),
        temp=None,
    ),
    TestedEntry(
        model_name=ModelName.Gemma3_27B,
        model_company=ModelCompany.Google,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "gemma-3-27b",
        ),
        temp=0.5,
    ),
    TestedEntry(
        model_name=ModelName.Gemma3_27B,
        model_company=ModelCompany.Google,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "gemma-3-27b",
        ),
        temp=0,
    ),
    TestedEntry(
        model_name=ModelName.Lingvanex,
        model_company=ModelCompany.Lingvanex,
        inference_service_name=InferenceCompany.Lingvanex,
        inference_source=LingvanexExecutableTranslator(LINGVANEX_API_KEY),
        temp=None,
    ),
    # TestedEntry(
    #    model_name=ModelName.Qwen_25_32B,
    #    model_company=ModelCompany.Alibaba,
    #    inference_service_name=InferenceCompany.Openrouter,
    #    inference_source=OpenrouterExecutableTranslator(
    #        OPENROUTER_API_KEY,
    #        "qwen-2-5-32b",
    #    ),
    # ),
    TestedEntry(
        model_name=ModelName.Qwen3_235b_a22b,
        model_company=ModelCompany.Alibaba,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "qwen-3-235b-a22b",
        ),
        temp=0.5,
    ),
    TestedEntry(
        model_name=ModelName.Qwen3_235b_a22b,
        model_company=ModelCompany.Alibaba,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "qwen-3-235b-a22b",
        ),
        temp=0,
    ),
    TestedEntry(
        model_name=ModelName.Qwen3_32b,
        model_company=ModelCompany.Alibaba,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "qwen-3-32b",
        ),
        temp=0.5,
    ),
    TestedEntry(
        model_name=ModelName.Qwen3_32b,
        model_company=ModelCompany.Alibaba,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "qwen-3-32b",
        ),
        temp=0,
    ),
    TestedEntry(
        model_name=ModelName.Qwen3_30_a3b,
        model_company=ModelCompany.Alibaba,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "qwen-3-30-a3b",
        ),
        temp=0.5,
    ),
    TestedEntry(
        model_name=ModelName.Qwen3_30_a3b,
        model_company=ModelCompany.Alibaba,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "qwen-3-30-a3b",
        ),
        temp=0.0,
    ),
    TestedEntry(
        model_name=ModelName.Qwen3_14b,
        model_company=ModelCompany.Alibaba,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "qwen-3-14b",
        ),
        temp=0.5,
    ),
    TestedEntry(
        model_name=ModelName.Qwen3_14b,
        model_company=ModelCompany.Alibaba,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "qwen-3-14b",
        ),
        temp=0,
    ),
    TestedEntry(
        model_name=ModelName.Llama_4_Maverick,
        model_company=ModelCompany.Meta,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "meta-llama/llama-4-maverick:free",
        ),
        temp=0.5,
    ),
    TestedEntry(
        model_name=ModelName.Llama_4_Maverick,
        model_company=ModelCompany.Meta,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "meta-llama/llama-4-maverick:free",
        ),
        temp=0,
    ),
    TestedEntry(
        model_name=ModelName.Llama_4_Scout,
        model_company=ModelCompany.Meta,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "llama-4-scout",
        ),
        temp=0.5,
    ),
    TestedEntry(
        model_name=ModelName.Llama_4_Scout,
        model_company=ModelCompany.Meta,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "llama-4-scout",
        ),
        temp=0,
    ),
    TestedEntry(
        model_name=ModelName.Llama33_70b,
        model_company=ModelCompany.Meta,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "llama-3.3-70b",
        ),
        temp=0.5,
    ),
    TestedEntry(
        model_name=ModelName.Llama33_70b,
        model_company=ModelCompany.Meta,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "llama-3.3-70b",
        ),
        temp=0,
    ),
    TestedEntry(
        model_name=ModelName.GPT_41_Nano,
        model_company=ModelCompany.OpenAI,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "gpt-4.1-nano",
        ),
        temp=0.5,
    ),
    TestedEntry(
        model_name=ModelName.GPT_41_Nano,
        model_company=ModelCompany.OpenAI,
        inference_service_name=InferenceCompany.Openrouter,
        inference_source=OpenrouterExecutableTranslator(
            OPENROUTER_API_KEY,
            "gpt-4.1-nano",
        ),
        temp=0,
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
    # TestedEntry(
    #    model_name=ModelName.Mistral_Small_Latest,
    #    model_company=ModelCompany.Mistral,
    #    inference_service_name=InferenceCompany.Mistral,
    #    inference_source=MistralExecutableTranslator(
    #        MISTRAL_API_KEY,
    #        "mistral-small-latest",
    #    ),
    # ),
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
        "google/gemini-2.5-pro-preview-03-25-comparison-system",
        OpenrouterGenericInference(
            OPENROUTER_API_KEY, "google/gemini-2.5-pro-preview-03-25"
        ),
    ),
]

temps_to_test = [0, 0.5]

# generate permutations to compare
# this generates "Compare sets". Each compare set will compare two models at a given language,
# using different temperatures and noting the type of sentence. Because _every compare set is identical_,
# and it is the smallest unit of comparing, this means we can also do analysis of different temperatures and
# sentence types!

# we generate the comparisons through:
# - every element to the one before and after it
# - every 4 elements are paired, so
# ...x...x...x...x also get paired - this is sparse
# - every 9 elements are paired in the same way
# part of the reason we do this is so that we don't obliterate the cache when we add new ones

# because we're just testing
evaluation_targets = evaluation_targets[:3]

print(evaluation_targets)

pairwise_items = []
for a, b in zip(evaluation_targets, evaluation_targets[1:]):
    pairwise_items.append((a, b))


def pair_every(pairwise_items, start, interval):
    if len(pairwise_items) <= start:
        return

    last = evaluation_targets[start]

    for i, x in enumerate(evaluation_targets[start:]):
        if i % interval == 0:
            pairwise_items.append((last, x))
            last = x


pair_every(pairwise_items, 0, 6)
pair_every(pairwise_items, 3, 18)
pair_every(pairwise_items, 5, 24)

print(
    [
        (
            a.model_name.value + "--" + str(a.temp),
            b.model_name.value + "--" + str(b.temp),
        )
        for a, b in pairwise_items
    ]
)


def md5hash(text):
    return hashlib.md5(bytes(text, "utf-8")).hexdigest()


def deterministic_coin_flip(s: str) -> bool:
    h = hashlib.sha256(s.encode()).digest()
    return bool(h[0] & 1)


# run inference and scoring. Obviously heavily heavily cache!
def compare_set(language, model_a, model_b):
    print(language, model_a, model_b)
    # aforementioned smallest unit of comparing. Note:
    # we will merely store the results of compare_models here. Later on we'll apply the
    # "must be 3/4" test in order to control (as best we can) for model lineage
    # keeping the results raw here lets us *not* do that, though, so we can actually
    # analyse the biases of each model intentionally
    output = []
    for category in SENTENCES_LIST.keys():
        i = 0
        for sentence in SENTENCES_LIST[category]:
            print("Checking", language, category, sentence)

            i += 1
            if i > 5:
                import sys

                sys.exit()

            def get_translation_with_cache_check(model):
                cache_key = f"TRANSLATION Language:{language.value}|Model: {model.unique_id()}|Sentence category:{category}|Sentence md5:{md5hash(sentence)}"
                check = cache.get(cache_key)
                if check:
                    return check
                else:
                    print("Translating", cache_key)
                    start_t = time.time()
                    translation = model.inference_source.translate(
                        TranslatableLanguage.English, language, sentence, model.temp
                    )
                    end_t = time.time()
                    cache.set(cache_key, translation)
                    log_latency(model.unique_id(), category, sentence, end_t - start_t)
                    return translation

            model_a_translation = get_translation_with_cache_check(model_a)
            model_b_translation = get_translation_with_cache_check(model_b)

            for comparison_model, comparison_inference in compare_models:
                print("Comparison with ", comparison_model)

                # handle 483's (refusals). This feels like a better method than effectively allowing them to
                # veto translations (if we just skipped instead)
                a_refusal = "483" in model_a_translation
                b_refusal = "483" in model_b_translation

                if a_refusal and b_refusal:
                    print("Both refused!")
                    continue
                elif a_refusal:
                    output.append(
                        ComparisonItem(
                            language=language,
                            tested_entry_a=model_a,
                            tested_entry_b=model_b,
                            sentence=sentence,
                            sentence_category=category,
                            evaluating_model="483-skip",
                            entry_a_translation=model_a_translation,
                            entry_b_translation=model_b_translation,
                            a_success=False,
                            b_success=True,
                            identical=False,
                            evaluating_response="N/A",
                        )
                    )
                    print("Only A refused")
                    continue
                elif b_refusal:
                    output.append(
                        ComparisonItem(
                            language=language,
                            tested_entry_a=model_a,
                            tested_entry_b=model_b,
                            sentence=sentence,
                            sentence_category=category,
                            evaluating_model="483-skip",
                            entry_a_translation=model_a_translation,
                            entry_b_translation=model_b_translation,
                            a_success=True,
                            b_success=False,
                            identical=False,
                            evaluating_response="N/A",
                        )
                    )
                    print("Only B refused")
                    continue

                # another special case: If they're literally the same, just write down Identical
                if model_a_translation.strip() == model_b_translation.strip():
                    print("Identical-Skip")
                    output.append(
                        ComparisonItem(
                            language=language,
                            tested_entry_a=model_a,
                            tested_entry_b=model_b,
                            sentence=sentence,
                            sentence_category=category,
                            evaluating_model=comparison_model,
                            entry_a_translation=model_a_translation,
                            entry_b_translation=model_b_translation,
                            a_success=False,
                            b_success=False,
                            identical=True,
                            evaluating_response="Autoskipped",
                        )
                    )
                    continue

                print(model_a_translation, "::", model_b_translation)

                coin_flip_key = f"COIN FLIP Language:{language.value}|Sentence cat:{category}|Sentence md5:{md5hash(sentence)}|{comparison_model}"
                swap_a_b = deterministic_coin_flip(coin_flip_key)

                if swap_a_b:
                    model_a_translation, model_b_translation = (
                        model_b_translation,
                        model_a_translation,
                    )

                comparison_prompt = f"""You're a translation expert. Compare two translations of the same sentence.
Think step-by-step in **very short notes** (20–200 words), e.g. "A has better tone. B idiomatic. A grammar slip."
Evaluate:
- Accuracy (same meaning?)
- Formality consistency
- Idiomaticity (native phrasing?)
- Correct vocab
- Correct grammar
Compare A and B as you go. On a **new line**, write only `Translation A`, `Translation B`, or `Identical` to show which is better.
If they're identical, say that immediately, no analysis.

Original: ```{sentence}```
A: ```{model_a_translation}```
B: ```{model_b_translation}```"""

                for i in range(0, 4):
                    key = f"COMPARISON hash:{md5hash(comparison_prompt)}"
                    print("KEY", key)
                    comparison_data = None

                    if cache.get(key):
                        comparison_data = cache.get(key)
                        print("Cache hit")
                    else:
                        print("Inference for comparison")
                        comparison_data = comparison_inference.infer(
                            comparison_prompt, 0
                        )

                    print("Comparison data", comparison_data)
                    last_line_unswapped = (
                        comparison_data.strip().split("\n")[-1].strip()
                    )
                    print("Last line unswapped", last_line_unswapped)
                    if swap_a_b:
                        last_line = last_line_unswapped
                        last_line.replace("Translation A", "tempA!!!000")
                        last_line.replace("Translation B", "tempB!!!000")
                        last_line.replace("tempA!!!000", "Translation B")
                        last_line.replace("tempB!!!000", "Translation A")
                        print("Swapped: ", last_line)
                    else:
                        last_line = last_line_unswapped

                    hit_count = 0
                    if "Translation A" in last_line:
                        hit_count += 1
                    if "Translation B" in last_line:
                        hit_count += 1
                    if "Identical" in last_line:
                        hit_count += 1

                    if hit_count > 1:
                        # invalid!
                        print(
                            "Reports for multiple in response: " + str(comparison_data)
                        )
                        continue

                    if "Translation A" in last_line:
                        cache.set(key, comparison_data)
                        output.append(
                            ComparisonItem(
                                language=language,
                                tested_entry_a=model_a,
                                tested_entry_b=model_b,
                                sentence=sentence,
                                sentence_category=category,
                                evaluating_model=comparison_model,
                                entry_a_translation=model_a_translation,
                                entry_b_translation=model_b_translation,
                                a_success=True,
                                b_success=False,
                                identical=False,
                                evaluating_response=comparison_data,
                            )
                        )
                        break
                    elif "Translation B" in last_line:
                        cache.set(key, comparison_data)
                        output.append(
                            ComparisonItem(
                                language=language,
                                tested_entry_a=model_a,
                                tested_entry_b=model_b,
                                sentence=sentence,
                                sentence_category=category,
                                evaluating_model=comparison_model,
                                entry_a_translation=model_a_translation,
                                entry_b_translation=model_b_translation,
                                a_success=False,
                                b_success=True,
                                identical=False,
                                evaluating_response=comparison_data,
                            )
                        )
                        break
                    elif "Identical" in last_line:
                        cache.set(key, comparison_data)
                        output.append(
                            ComparisonItem(
                                language=language,
                                tested_entry_a=model_a,
                                tested_entry_b=model_b,
                                sentence=sentence,
                                sentence_category=category,
                                evaluating_model=comparison_model,
                                entry_a_translation=model_a_translation,
                                entry_b_translation=model_b_translation,
                                a_success=False,
                                b_success=False,
                                identical=True,
                                evaluating_response=comparison_data,
                            )
                        )
                        break
                    else:
                        print("Cannot get selection from: ", comparison_data)

    return output


compare_sets = []
i = 0
for language in target_languages:
    j = 0
    for model_a, model_b in pairwise_items:
        progress_num = (i * len(pairwise_items)) + j
        progress_denom = len(target_languages) * len(pairwise_items)
        progress_percent = (progress_num * 100) / progress_denom
        print(f"{progress_num}/{progress_denom} | {round(progress_percent, 2)}%")

        comparisons = compare_set(language, model_a, model_b)
        to_add = {
            "language": language,
            "model_a": model_a,
            "model_b": model_b,
            "comparisons": comparisons,
        }
        compare_sets.append(to_add)

        j += 1

    i += 1

# Now build the rankings...
for language in target_languages:
    # find comparisons
    relevant_comparisons = [x for x in compare_sets if x["language"] == language]

    # first: simple multi-unit consensus
    # let's produce a graph...

    # various variants we want to generate:
    # - what if [any combination] of judgers were turned on
    # - what if we required a consensus of (4, 3, 2) before we generated one "comparison" ("Single unit consensus")
    # - what if we keep the "low-res comparisons", but only activated if there's over (4, 3, 2) consensus ("Multi-unit consensus")

    # ALL of that, but sorted by sentence types (yeah we're going to want to make it modular)
