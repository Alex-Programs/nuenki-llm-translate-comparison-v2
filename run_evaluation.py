import hashlib
import time
from dataset import SENTENCES_LIST
from typedefinitions import *
from davidson_model import DavidsonBT
from latency_logger import log_latency


def pair_every(pairwise_items, input_data, start, interval, limit):
    if len(input_data) <= start:
        return

    last = input_data[start]

    count = 0
    for i, x in enumerate(input_data[start:]):
        if count >= limit:
            break

        if i % interval == 0 and last != x:
            pairwise_items.append((last, x))
            last = x
            count += 1


def md5hash(text):
    return hashlib.md5(bytes(text, "utf-8")).hexdigest()


def deterministic_coin_flip(s: str) -> bool:
    h = hashlib.sha256(s.encode()).digest()
    return bool(h[0] & 1)


def compare_set(language, model_a, model_b, cache, compare_models):
    print(language, model_a, model_b)
    if model_a == model_b:
        print("CANNOT TEST MODELS AGAINST THEMSELVES!")
        import sys

        sys.exit()
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
                            b_success=False,
                            identical=False,
                            evaluating_response="N/A",
                        )
                    )
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

                coin_flip_key = f"COIN FLIP Language:{language.value}|Sentence cat:{category}|Sentence md5:{md5hash(sentence)}|{comparison_model}|{model_a.unique_id()}|{model_b.unique_id()}"
                swap_a_b = deterministic_coin_flip(coin_flip_key)

                if swap_a_b:
                    print("SWAPPING ORDER!!")
                    model_a_translation, model_b_translation = (
                        model_b_translation,
                        model_a_translation,
                    )

                comparison_prompt = f"""You're a translation expert. Compare two translations of the same sentence.
Your gold standard is the idiomatic, native, absolutely correct style found in a high‑quality language‑learning textbook.
Think step-by-step in **VERY** short notes (20–200 words), e.g. "A is more accurate. B uses more idiomatic phrasing. B uses 'Schlecht' incorrectly." Nobody will read them, so don't use fancy styling: They are merely your own chain of thought.
Evaluate, in approx. priority high-to-low:
- Accuracy (Same meaning conferred?)
- Idiomaticity & style (native phrasing? Matches the desired style?)
- Correct vocab
- Correct grammar
- Consistency in formality
Compare A and B as you go. On a **new line**, write only `Translation A`, `Translation B`, or `Identical` to show which is better.

Original: ```{sentence}```
A: ```{model_a_translation}```
B: ```{model_b_translation}```"""

                for i in range(0, 4):
                    key = f"COMPARISON hash:{md5hash(comparison_prompt)} Model:{comparison_model}"
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
                        last_line = last_line.replace("Translation A", "tempA!!!000")
                        last_line = last_line.replace("Translation B", "tempB!!!000")
                        last_line = last_line.replace("tempA!!!000", "Translation B")
                        last_line = last_line.replace("tempB!!!000", "Translation A")
                        print("Swapped: ", last_line)

                        model_a_translation, model_b_translation = (
                            model_b_translation,
                            model_a_translation,
                        )
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

    pair_every(pairwise_items, target_models, 3, 5, 999)
    pair_every(pairwise_items, target_models, 0, 6, 999)
    pair_every(pairwise_items, target_models, 3, 18, 999)
    pair_every(pairwise_items, target_models, 5, 24, 999)
    pair_every(pairwise_items, target_models, 0, 2, 4)

    print("Pairwise length before dedup", len(pairwise_items))
    dedup_pairwise = []
    for x in pairwise_items:
        if x in dedup_pairwise:
            continue
        dedup_pairwise.append(x)

    pairwise_items = dedup_pairwise
    print("Pairwise length after dedup", len(pairwise_items))

    print("Got pairwise items: ", pairwise_items)
    for a, b in pairwise_items:
        print(f"{a.model_name}:{a.temp} || {b.model_name}:{b.temp}")

    compare_sets = []

    i = 0
    for language in target_languages:
        j = 0
        for model_a, model_b in pairwise_items:
            progress_num = (i * len(pairwise_items)) + j
            progress_denom = len(target_languages) * len(pairwise_items)
            progress_percent = (progress_num * 100) / progress_denom
            print(f"{progress_num}/{progress_denom} | {round(progress_percent, 2)}%")

            print("MODEL A: ", model_a)
            print("MODEL B: ", model_b)
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

        comparison_items = []
        for comparison_set in relevant_comparison_sets:
            for comparison in comparison_set["comparisons"]:
                comparison_items.append(comparison)

        model = produce_model_from_dataset(id_model_pairs, comparison_items)

        scores = product_rankings_from_strengths(id_model_pairs, model.get_strengths())
        print_scores(scores)

        ### AMPLIFICATION VARIANT ###
        comparison_items_amplified_synthetic = amplify(
            relevant_comparison_sets, language
        )
        model_amplified = produce_model_from_dataset(
            id_model_pairs, comparison_items_amplified_synthetic
        )
        scores = product_rankings_from_strengths(
            id_model_pairs, model_amplified.get_strengths()
        )
        print("Amplified")
        print_scores(scores)

        # various variants we want to generate:
        # - standard with signal amplification and noise reduction: yes, we do keep individual comparison, but
        # where there is consensus we amplify by creating new ones, and when there is not we dampen. In order to do this,
        # scale up the "unit of comparison" to 3.

        # - what if [any combination] of judgers were turned on - with and without amplification
        # - what if we required a consensus of (4, 3) before we generated one "comparison" ("Single unit consensus")
        # - what if we keep the "high-res comparisons", but only activated if there's over (4, 3, 2) consensus ("Multi-unit consensus")

        # ALL of that, but sorted by sentence types (yeah we're going to want to make it modular)


def amplify(relevant_comparison_sets, language):
    synthetic_comparison_items_amplified = []
    for comparison_set in relevant_comparison_sets:
        grouped_equal_comparisons = {}
        for comparison in comparison_set["comparisons"]:
            key = (
                comparison.sentence
                + "|"
                + comparison.tested_entry_a.unique_id()
                + "|"
                + comparison.tested_entry_b.unique_id()
            )
            if key not in grouped_equal_comparisons.keys():
                grouped_equal_comparisons[key] = []

            grouped_equal_comparisons[key].append(comparison)

        for key, grouped in grouped_equal_comparisons.items():
            a_quant = len([x for x in grouped if x.a_success])
            b_quant = len([x for x in grouped if x.b_success])
            ident_quant = len([x for x in grouped if x.identical])
            print(a_quant, b_quant, ident_quant)

            model_a = grouped[0].tested_entry_a
            model_b = grouped[0].tested_entry_b
            sentence = grouped[0].sentence
            sentence_category = grouped[0].sentence_category

            synthetic = ComparisonItem(
                language=language,
                tested_entry_a=model_a,
                tested_entry_b=model_b,
                sentence=sentence,
                sentence_category=sentence_category,
                evaluating_model="Synthetic Amplified",
                entry_a_translation="N/A",
                entry_b_translation="N/A",
                a_success=False,
                b_success=False,
                identical=False,
                evaluating_response="N/A",
            )

            # https://www.desmos.com/3d/mdpijejest
            if a_quant == b_quant or (a_quant == 0 and b_quant == 0):
                # inject a "same" one, based on the amount of "ident"
                ident_score = ident_quant**2

                synthetic.identical = True

                for _ in range(ident_score):
                    synthetic_comparison_items_amplified.append(synthetic)

                continue

            a_consensus_score = a_quant**2 - b_quant**2
            b_consensus_score = b_quant**2 - a_quant**2

            # there will be no equals - due to checking earlier
            if a_consensus_score > b_consensus_score:
                synthetic.a_success = True
            else:
                synthetic.b_success = True

            for _ in range(max(a_consensus_score, b_consensus_score)):
                synthetic_comparison_items_amplified.append(synthetic)

    return synthetic_comparison_items_amplified


def produce_model_from_dataset(id_model_pairs, actual_comparisons):
    comparisons_for_input = []
    o = 0
    p = 0
    for item in actual_comparisons:
        id_a = None
        id_b = None

        for id, model in id_model_pairs:
            if model == item.tested_entry_a:
                id_a = id
            elif model == item.tested_entry_b:
                id_b = id

        if item.a_success:
            comparisons_for_input.append((id_a, id_b, "win1"))
        elif item.b_success:
            comparisons_for_input.append((id_a, id_b, "win2"))
        elif item.identical:
            comparisons_for_input.append((id_a, id_b, "tie"))
            p += 1

        o += 1

    print(p, o)

    model = DavidsonBT.from_comparisons(comparisons_for_input)
    return model


def product_rankings_from_strengths(id_model_pairs, strengths):
    i = 0
    scores = []
    for strength in strengths:
        corresponding = [b for a, b in id_model_pairs if a == i][0]
        scores.append((strength, corresponding))

        i += 1

    scores_sorted = sorted(scores, key=lambda x: x[0], reverse=True)
    return scores_sorted


def print_scores(scores):
    for score, item in scores:
        print("----------" * 4)
        print(
            f"{score}:\t\t{item.model_name}:{item.temp} ({item.model_company} via {item.inference_source})"
        )
