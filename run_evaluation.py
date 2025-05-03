import hashlib
import time
from dataset import SENTENCES_LIST
from typedefinitions import *

from latency_logger import log_latency


def pair_every(pairwise_items, start, interval):
    if len(pairwise_items) <= start:
        return

    last = pairwise_items[start]

    for i, x in enumerate(pairwise_items[start:]):
        if i % interval == 0:
            pairwise_items.append((last, x))
            last = x


def md5hash(text):
    return hashlib.md5(bytes(text, "utf-8")).hexdigest()


def deterministic_coin_flip(s: str) -> bool:
    h = hashlib.sha256(s.encode()).digest()
    return bool(h[0] & 1)


def compare_set(language, model_a, model_b, cache, compare_models):
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


def evaluate_datasets(target_languages, target_models, cache, compare_models):
    pairwise_items = []

    for a, b in zip(target_models, target_models[1:]):
        pairwise_items.append((a, b))

    pair_every(pairwise_items, 0, 6)
    pair_every(pairwise_items, 3, 18)
    pair_every(pairwise_items, 5, 24)

    print("Got pairwise items: ", pairwise_items)

    compare_sets = []

    i = 0
    for language in target_languages:
        j = 0
        for model_a, model_b in pairwise_items:
            progress_num = (i * len(pairwise_items)) + j
            progress_denom = len(target_languages) * len(pairwise_items)
            progress_percent = (progress_num * 100) / progress_denom
            print(f"{progress_num}/{progress_denom} | {round(progress_percent, 2)}%")

            comparisons = compare_set(language, model_a, model_b, cache, compare_models)
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
        relevant_comparison_sets = [
            x for x in compare_sets if x["language"] == language
        ]

        # first: simple multi-unit consensus
        # let's produce a graph...
        id_model_pairs = []
        current_id = 0

        for comparison_set in relevant_comparison_sets:
            existing_models = [b for a, b in id_model_pairs]
            if comparison_set["model_a"] not in existing_models:
                id_model_pairs.append((current_id, comparison_set["model_a"]))
                current_id += 1

            if comparison_set["model_b"] not in existing_models:
                id_model_pairs.append((current_id, comparison_set["model_b"]))
                current_id += 1

        print("ID-model pairs: ", id_model_pairs)

        # various variants we want to generate:
        # - what if [any combination] of judgers were turned on
        # - what if we required a consensus of (4, 3, 2) before we generated one "comparison" ("Single unit consensus")
        # - what if we keep the "low-res comparisons", but only activated if there's over (4, 3, 2) consensus ("Multi-unit consensus")

        # ALL of that, but sorted by sentence types (yeah we're going to want to make it modular)
