import hashlib
import time
from dataset import SENTENCES_LIST
from typedefinitions import *
from davidson_model import DavidsonBT
from latency_logger import log_latency
from itertools import combinations
import threading
from queue import Queue
from random import choice
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr, pearsonr  # Ensure scipy is installed


def all_non_duplicate_sets(input_list):
    result = []
    for r in range(1, len(input_list) + 1):
        result.extend([list(c) for c in combinations(input_list, r)])
    return result


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


def log_sentence_data(*args, **kwargs):
    return
    print("S", *args, **kwargs)


def process_sentence(
    language, category, sentence, model_a, model_b, cache, compare_models, output_queue
):
    log_sentence_data("Checking", language, category, sentence)

    def get_translation_with_cache_check(model):
        cache_key = f"TRANSLATION Language:{language.value}|Model: {model.unique_id()}|Sentence category:{category}|Sentence md5:{md5hash(sentence)}"
        check = cache.get(cache_key)
        if check:
            return check
        else:
            log_sentence_data("Translating", cache_key)

            # this is to avoid getting ratelimited...
            sleep_t = choice(range(0, 4))
            time.sleep(sleep_t)

            start_t = None
            end_t = None
            translation = None

            i = 0
            while True:
                log_sentence_data("Translation attempt", i)
                i += 1
                try:
                    start_t = time.time()
                    translation = model.inference_source.translate(
                        TranslatableLanguage.English, language, sentence, model.temp
                    )
                    end_t = time.time()
                    break
                except Exception as e:
                    log_sentence_data("Err on translate", e, "with m", model)
                    time.sleep(i * choice(range(3, 12)))

            cache.set(cache_key, translation)
            log_latency(model.unique_id(), category, sentence, end_t - start_t)
            return translation

    model_a_translation = get_translation_with_cache_check(model_a)
    model_b_translation = get_translation_with_cache_check(model_b)

    for comparison_model, comparison_inference in compare_models:
        log_sentence_data("MMM", model_a, model_b)
        log_sentence_data("Comparison with ", comparison_model)

        # handle 483's (refusals). This feels like a better method than effectively allowing them to
        # veto translations (if we just skipped instead)
        a_refusal = "483" in model_a_translation
        b_refusal = "483" in model_b_translation

        if a_refusal and b_refusal:
            output_queue.put(
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
            log_sentence_data("Both refused!")
            continue
        elif a_refusal:
            output_queue.put(
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
            log_sentence_data("Only A refused")
            continue
        elif b_refusal:
            output_queue.put(
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
            log_sentence_data("Only B refused")
            continue

        # another special case: If they're literally the same, just write down Identical
        if model_a_translation.strip() == model_b_translation.strip():
            log_sentence_data("Identical-Skip")
            output_queue.put(
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

        log_sentence_data(model_a_translation, "::", model_b_translation)

        coin_flip_key = f"COIN FLIP Language:{language.value}|Sentence cat:{category}|Sentence md5:{md5hash(sentence)}|{comparison_model}|{model_a.unique_id()}|{model_b.unique_id()}"
        swap_a_b = deterministic_coin_flip(coin_flip_key)

        if swap_a_b:
            log_sentence_data("SWAPPING ORDER!!")
            model_a_translation, model_b_translation = (
                model_b_translation,
                model_a_translation,
            )

        comparison_prompt = f"""You're an unforgiving professional translator. Compare two translations of the same sentence.
IGNORE subjective or arguable style differences (e.g. using a common or technical English loan word; passive vs active voice). If a choice can be reasonably defended as a subjective but valid choice, do not criticise it. Prefer natural translations over literal ones that sound non-native.

Think step-by-step in *only a few words per point*, like "A better at 'Hallo'. B more idiomatic. B misuses 'Schlecht'." AVOID VERBOSITY. Just terse, for yourself. Save tokens.

Evaluate in order of priority (high-to-low):
- Accuracy (Same meaning?)
- Vocab (Check for any mistakes, however subtle)
- Grammar accuracy
- Tone (does it match the original?)
- *Consistency* in formality, both within the translation and compared to the original
- Idiomaticity & style (native phrasing? Remember, no stylistic judgements)
Remember priorities when tiebreaking.

If equally good (unable to decide) or identical, say `Identical`. On a **new line**, output ONLY:
`Translation A`, `Translation B`, or `Identical`.

Original: ```{sentence}```
A: ```{model_a_translation}```
B: ```{model_b_translation}```"""

        key = f"COMPARISON hash:{md5hash(comparison_prompt)} Model:{comparison_model}"
        log_sentence_data("KEY", key)
        comparison_data = None

        if cache.get(key):
            comparison_data = cache.get(key)
            log_sentence_data("Cache hit")
        else:
            log_sentence_data("Inference for comparison")
            wait_t = choice(range(0, 6))
            time.sleep(wait_t)

            for i in range(3):
                try:
                    comparison_data = comparison_inference.infer(comparison_prompt, 0)
                    if comparison_data:
                        break
                except Exception as e:
                    log_sentence_data(
                        "Error during inference:",
                        e,
                        "with comp model",
                        comparison_model,
                    )
                    time.sleep(i * choice(range(5, 25)))

        if not comparison_data:
            continue

        log_sentence_data("Comparison data", comparison_data)
        last_line_unswapped = comparison_data.strip().split("\n")[-1].strip()
        if (
            not "Translation" in last_line_unswapped
            and not "Identical" in last_line_unswapped
            and len(last_line_unswapped.split("\n")) > 2
        ):
            penul = comparison_data.strip().split("\n")[-2].strip()
            if (
                "Translation" in penul
                or "Identical" in penul
                and "(Note:" in last_line_unswapped
            ):
                # fix gemini being stupid without a rerun
                last_line_unswapped = penul
            else:
                log_sentence_data(
                    f"TRY REPLACE: LAST LINE UNSWAPPED BEFORE: `{last_line_unswapped}`"
                )
                # try to fix it this way...
                last_line_unswapped = last_line_unswapped.replace(
                    "B.", "Translation B"
                ).replace("A.", "Translation A")

                if last_line_unswapped == "A":
                    last_line_unswapped = "Translation A"
                elif last_line_unswapped == "B":
                    last_line_unswapped = "Translation B"

                log_sentence_data(
                    f"TRY REPLACE: LAST LINE UNSWAPPED AFTER: `{last_line_unswapped}`"
                )

        log_sentence_data("Last line unswapped", last_line_unswapped)
        if swap_a_b:
            last_line = last_line_unswapped
            last_line = last_line.replace("Translation A", "tempA!!!000")
            last_line = last_line.replace("Translation B", "tempB!!!000")
            last_line = last_line.replace("tempA!!!000", "Translation B")
            last_line = last_line.replace("tempB!!!000", "Translation A")
            log_sentence_data("Swapped: ", last_line)

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
            log_sentence_data(
                "Reports for multiple in response: " + str(comparison_data)
            )
            continue

        if "Translation A" in last_line:
            cache.set(key, comparison_data)
            output_queue.put(
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
            continue
        elif "Translation B" in last_line:
            cache.set(key, comparison_data)
            output_queue.put(
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
            continue
        elif "Identical" in last_line:
            cache.set(key, comparison_data)
            output_queue.put(
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
            continue
        else:
            log_sentence_data("Cannot get selection from: ", comparison_data)


def compare_set(language, model_a, model_b, cache, compare_models):
    print(language, model_a, model_b)
    if model_a == model_b:
        print("CANNOT TEST MODELS AGAINST THEMSELVES!")
        import sys

        sys.exit()

    output = []
    threads = []
    output_queue = Queue()

    for category in SENTENCES_LIST.keys():
        for sentence in SENTENCES_LIST[category]:
            thread = threading.Thread(
                target=process_sentence,
                args=(
                    language,
                    category,
                    sentence,
                    model_a,
                    model_b,
                    cache,
                    compare_models,
                    output_queue,
                ),
            )
            threads.append(thread)
            thread.start()

    for thread in threads:
        thread.join()

    while not output_queue.empty():
        output.append(output_queue.get())

    return output


def evaluate_datasets(target_languages, target_models, cache, compare_models):
    pairwise_items = []

    for a, b in zip(target_models, target_models[1:]):
        pairwise_items.append((a, b))

    # pair_every(pairwise_items, target_models, 3, 5, 999)
    # pair_every(pairwise_items, target_models, 0, 6, 999)
    # pair_every(pairwise_items, target_models, 3, 18, 999)
    # pair_every(pairwise_items, target_models, 5, 24, 999)
    # pair_every(pairwise_items, target_models, 0, 2, 4)
    # pair_every(pairwise_items, target_models, 1, 3, 6)

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

    from concurrent.futures import ThreadPoolExecutor, as_completed

    compare_sets = []

    pairwise_log = []

    for i, language in enumerate(target_languages):
        futures = []
        with ThreadPoolExecutor() as executor:
            for j, (model_a, model_b) in enumerate(pairwise_items):
                progress_num = (i * len(pairwise_items)) + j
                progress_denom = len(target_languages) * len(pairwise_items)
                progress_percent = (progress_num * 100) / progress_denom
                print(
                    f"{progress_num}/{progress_denom} | {round(progress_percent, 2)}%"
                )
                print("MODEL A: ", model_a)
                print("MODEL B: ", model_b)

                futures.append(
                    executor.submit(
                        compare_set, language, model_a, model_b, cache, compare_models
                    )
                )

            for j, (future, (model_a, model_b)) in enumerate(
                zip(futures, pairwise_items)
            ):
                comparisons = future.result()
                to_add = {
                    "language": language,
                    "model_a": model_a,
                    "model_b": model_b,
                    "comparisons": comparisons,
                }
                compare_sets.append(to_add)
                pairwise_log.append((model_a, model_b))

    print("Finished initial comparisons")
    # iterative active learning loop using entropy

    MIN_ENTROPY = 1.09054

    print(len(pairwise_log))

    while True:
        pairs_info = []  # (language, model_a, model_b)

        for language in target_languages:
            print("Target-iteration over", language)

            relevant_sets = [x for x in compare_sets if x["language"] == language]
            id_model_pairs = produce_id_model_pairs(relevant_sets)

            comparison_items = []
            for cs in relevant_sets:
                comparison_items.extend(cs["comparisons"])

            print("Fitting...")
            model_base = produce_model_from_dataset(id_model_pairs, comparison_items)
            print("Fit iterative model")

            # map id -> model
            id_to_model = dict(id_model_pairs)

            # pick highest-entropy untested pair
            print(model_base.rank_pairwise_by_entropy())

            for id_a, id_b, entropy in model_base.rank_pairwise_by_entropy():
                print("Ent", entropy)
                if entropy < MIN_ENTROPY:
                    print(
                        "Skipping",
                        id_a,
                        id_b,
                        "due to entropy",
                        entropy,
                        "below minimum",
                        MIN_ENTROPY,
                    )
                    continue

                model_a = id_to_model[id_a]
                model_b = id_to_model[id_b]
                if (model_a, model_b) in pairwise_log or (
                    model_b,
                    model_a,
                ) in pairwise_log:
                    print("Skipping due to pair already done")
                    continue

                print(
                    f"Targeting entropy {entropy:.3f} between {model_a} and {model_b}"
                )
                pairs_info.append((language, model_a, model_b))
                break

        print("Pairs info", pairs_info)
        if not pairs_info:
            # We're done!
            break

        for language, model_a, model_b in pairs_info:
            print("Comparing", model_a, model_b)
            print("Already done", len(pairwise_log), "comparisons")
            resp = compare_set(language, model_a, model_b, cache, compare_models)
            print("Comparison done")
            compare_sets.append(
                {
                    "language": language,
                    "model_a": model_a,
                    "model_b": model_b,
                    "comparisons": resp,
                }
            )
            pairwise_log.append((model_a, model_b))

    print(len(pairwise_log))

    # Now build the rankings...
    langs_results = []
    for language in target_languages:
        # find comparisons
        relevant_comparison_sets = [
            x for x in compare_sets if x["language"] == language
        ]

        # first: simple multi-unit consensus
        # let's produce a graph...
        id_model_pairs = produce_id_model_pairs(relevant_comparison_sets)
        print("ID-model pairs: ", id_model_pairs)

        comparison_items = []
        for comparison_set in relevant_comparison_sets:
            for comparison in comparison_set["comparisons"]:
                comparison_items.append(comparison)

        ### MOST STANDARD VARIANT ###
        model_base = produce_model_from_dataset(id_model_pairs, comparison_items)
        base_data = produce_sane_data_from_model(id_model_pairs, model_base)
        print(base_data)

        ### AMPLIFICATION VARIANT ###
        comparison_items_amplified_synthetic = amplify(comparison_items)
        model_amplified = produce_model_from_dataset(
            id_model_pairs, comparison_items_amplified_synthetic
        )
        amplified_data = produce_sane_data_from_model(id_model_pairs, model_base)
        print(amplified_data)
        print_model_rankings(amplified_data)

        judge_names = [x for x, y in compare_models]

        out = {
            "judges": judge_names,
            "base": base_data,
            "amplified": amplified_data,
            "judges_comb": [],
            "sentences_comb": [],
        }

        judge_combs = all_non_duplicate_sets(judge_names)

        for comb in judge_combs:
            dataset = [x for x in comparison_items if x.evaluating_model in comb]
            model_base = produce_model_from_dataset(id_model_pairs, dataset)
            sane_base = produce_sane_data_from_model(id_model_pairs, model_base)

            amplified_synthetic = amplify(dataset)
            model_synthetic = produce_model_from_dataset(
                id_model_pairs, amplified_synthetic
            )
            sane_amplified = produce_sane_data_from_model(
                id_model_pairs, model_synthetic
            )
            out["judges_comb"].append(
                {
                    "judges": comb,
                    "base_data": sane_base,
                    "amplified_data": sane_amplified,
                }
            )

        sentence_combs = all_non_duplicate_sets(SENTENCES_LIST.keys())
        for comb in sentence_combs:
            dataset = [x for x in comparison_items if x.sentence_category in comb]
            model_base = produce_model_from_dataset(id_model_pairs, dataset)
            sane_base = produce_sane_data_from_model(id_model_pairs, model_base)

            amplified_synthetic = amplify(dataset)
            model_synthetic = produce_model_from_dataset(
                id_model_pairs, amplified_synthetic
            )
            sane_amplified = produce_sane_data_from_model(
                id_model_pairs, model_synthetic
            )
            out["sentences_comb"].append(
                {
                    "judges": comb,
                    "base_data": sane_base,
                    "amplified_data": sane_amplified,
                }
            )

        print_consensus_score_by_sentence_type(comparison_set["comparisons"])

        langs_results.append({"language": language.value, "data": out})

        print_model_rankings(amplified_data)

        # various variants we want to generate:
        # - standard with signal amplification and noise reduction: yes, we do keep individual comparison, but
        # where there is consensus we amplify by creating new ones, and when there is not we dampen. In order to do this,
        # scale up the "unit of comparison" to 3.

        # - what if [any combination] of judgers were turned on - with and without amplification
        # - what if we required a consensus of (4, 3) before we generated one "comparison" ("Single unit consensus")
        # - what if we keep the "high-res comparisons", but only activated if there's over (4, 3, 2) consensus ("Multi-unit consensus")

        # ALL of that, but sorted by sentence types (yeah we're going to want to make it modular)

        datadump = analyze_category_signal_strength(comparison_items, SENTENCES_LIST)

        print_category_signal_analysis(datadump)

        print_judge_analysis(
            analyze_judge_performance(
                comparison_items,
                target_models,
                [name_tuple[0] for name_tuple in compare_models],
            )
        )

    return langs_results


def produce_id_model_pairs(relevant_comparison_sets):
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

    return id_model_pairs


def print_consensus_score_by_sentence_type(comparisons):
    grouped_equal_comparisons = produce_grouped_equal_comparisons(comparisons)
    for sentence_type in SENTENCES_LIST.keys():
        print("Sentence Type:", sentence_type)
        num = 0
        denom = 0
        for equal_comp in grouped_equal_comparisons.values():
            if equal_comp[0].sentence_category == sentence_type:
                amount_a = len([x for x in equal_comp if x.a_success])
                amount_b = len([x for x in equal_comp if x.b_success])
                amount_identical = len([x for x in equal_comp if x.identical])

                consensus_score = abs(amount_a - amount_b) - (amount_identical * 0.5)
                print(consensus_score)
                num += consensus_score
                denom += 1

        print("Avg", str(num * 100 / denom))


def produce_grouped_equal_comparisons(comparisons):
    grouped_equal_comparisons = {}
    for comparison in comparisons:
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

    return grouped_equal_comparisons


def amplify(relevant_comparisons):
    synthetic_comparison_items_amplified = []
    grouped_equal_comparisons = produce_grouped_equal_comparisons(relevant_comparisons)

    for key, grouped in grouped_equal_comparisons.items():
        a_quant = len([x for x in grouped if x.a_success])
        b_quant = len([x for x in grouped if x.b_success])
        ident_quant = len([x for x in grouped if x.identical])

        model_a = grouped[0].tested_entry_a
        model_b = grouped[0].tested_entry_b
        sentence = grouped[0].sentence
        sentence_category = grouped[0].sentence_category
        lang = grouped[0].language

        synthetic = ComparisonItem(
            language=lang,
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


def produce_rankings_from_strengths(id_model_pairs, strengths):
    i = 0
    scores = []
    for strength in strengths:
        corresponding = [b for a, b in id_model_pairs if a == i][0]
        scores.append((strength, corresponding))

        i += 1

    scores_sorted = sorted(scores, key=lambda x: x[0], reverse=True)
    return scores_sorted


def produce_sane_data_from_model(id_model_pairs, model):
    sane_models_storage = []
    strengths_intervals = model.get_confidence_intervals()

    i = 0
    for low, mid, high in strengths_intervals:
        corresponding = [b for a, b in id_model_pairs if a == i][0]
        sane_models_storage.append(
            {
                "model": corresponding.dump_data(),
                "model_id": i,
                "intervals_95": {
                    "low": round(low, 4),
                    "mid": round(mid, 4),
                    "high": round(high, 4),
                },
                "p_vals": {},
            }
        )

        i += 1

    sane_models_storage = sorted(
        sane_models_storage, key=lambda x: x["intervals_95"]["mid"]
    )

    p_vals = model.pairwise_p_values_full()

    for (k, v), p in p_vals.items():
        for m in sane_models_storage:
            if m["model_id"] == k:
                m["p_vals"][v] = round(p, 4)

    return sane_models_storage


def print_model_rankings(ranked_models_data, language_name="Overall"):
    """
    Prints a best-to-worst leaderboard of models based on their performance scores
    (strengths) and 95% confidence intervals.

    Args:
        ranked_models_data: A list of dictionaries, where each dictionary contains
                            model information and performance metrics. This is typically
                            the output of `produce_sane_data_from_model`.
        language_name: Optional name of the language or context for which rankings
                       are being printed. Defaults to "Overall".
    """
    if not ranked_models_data:
        print(f"No model data available to print rankings for {language_name}.")
        return

    # The input `ranked_models_data` from `produce_sane_data_from_model`
    # is sorted by 'mid' score in ascending order.
    # For a best-to-worst leaderboard (highest score first), we re-sort or reverse.
    # Creating a new sorted list for clarity and to avoid modifying the input list.
    sorted_for_leaderboard = sorted(
        ranked_models_data, key=lambda x: x["intervals_95"]["mid"], reverse=True
    )

    print(f"\n--- Model Rankings for {language_name} ---")
    print("=" * 90)  # Adjusted total width
    # Header for the leaderboard table
    # Right-aligning 'Strength' for better numerical readability
    header = (
        f"{'Rank':<5} {'Model Name':<35} {'Temp':<6} {'Strength':>10} "
        f"{'95% CI (Strength)':<25}"  # Adjusted width for CI
    )
    print(header)
    print("-" * 90)  # Adjusted total width

    for i, item in enumerate(sorted_for_leaderboard):
        rank = i + 1
        model_info = item.get("model", {})  # Safely get model_info

        # Extract model name and temperature
        # .dump_data() from ModelEntry should provide these
        model_name = model_info.get("model_name", "N/A")
        model_temp = model_info.get("temp", "N/A")

        intervals = item.get("intervals_95", {})  # Safely get intervals
        score_mid = intervals.get("mid", float("nan"))  # Use NaN for missing data
        score_low = intervals.get("low", float("nan"))
        score_high = intervals.get("high", float("nan"))

        # Format the output string for each model
        # Using .2f for float formatting, ensures two decimal places
        # Temp is converted to string to handle various types (float, "N/A")
        # CI string is formatted for clarity
        ci_string = f"[{score_low:.2f} - {score_high:.2f}]"

        row = (
            f"{rank:<5} {model_name:<35} {str(model_temp):<6} {score_mid:>10.2f} "
            f"{ci_string:<25}"
        )
        print(row)

    print("=" * 90)  # Adjusted total width
    print(
        "\nNote: 'Strength' refers to the estimated model ability from the Davidson-Bradley-Terry model."
    )
    print("Higher strength values indicate better performance in pairwise comparisons.")
    print(
        "95% CI indicates the range within which the true strength likely lies with 95% confidence."
    )


# Temporary, LLM-generated, for some NON-RIGOROUS analysis into different sentence types


import numpy as np
from scipy.stats import (
    norm,
)  # Make sure this is imported if not already at the top of your main script
from collections import defaultdict

# Assuming your SENTENCES_LIST, ComparisonItem, DavidsonBT,
# produce_id_model_pairs, produce_model_from_dataset (for its comparison transformation part)
# are defined as in your provided code.


def analyze_category_signal_strength(
    all_comparison_items: list[ComparisonItem],
    sentences_data: dict[str, list[str]],  # This is your SENTENCES_LIST
    min_comparisons_for_bt: int = 10,  # Min comparisons needed in a category for BT analysis
    min_models_for_bt: int = 3,  # Min unique models involved for BT analysis
    p_value_threshold: float = 0.05,  # For counting significant differences
):
    """
    Analyzes each sentence category to determine its signal strength for model evaluation.

    Args:
        all_comparison_items: A list of all ComparisonItem objects from evaluations.
        sentences_data: The SENTENCES_LIST dictionary mapping category names to sentences.
        min_comparisons_for_bt: Minimum comparisons for fitting a category-specific BT model.
        min_models_for_bt: Minimum unique models involved in a category for BT model.
        p_value_threshold: Alpha level for considering a p-value significant.

    Returns:
        A list of dictionaries, each containing metrics for a category,
        sorted by a composite signal score (higher is better).
    """
    category_metrics = []

    for category_name in sentences_data.keys():
        print(f"\n--- Analyzing Category: {category_name} ---")

        # 1. Filter comparisons for this category
        category_comparisons = [
            item
            for item in all_comparison_items
            if item.sentence_category == category_name
        ]

        if not category_comparisons:
            print(f"No comparisons found for category: {category_name}. Skipping.")
            category_metrics.append(
                {
                    "category": category_name,
                    "num_total_comparisons_in_log": 0,
                    "judge_agreement_score": 0,
                    "decisiveness_score": 0,
                    "bt_strength_spread": 0,
                    "bt_avg_ci_width": float("inf"),
                    "bt_significant_pairs_ratio": 0,
                    "bt_model_fitted": False,
                    "composite_signal_score": 0,
                    "notes": "No comparisons in log for this category.",
                }
            )
            continue

        num_total_comparisons_in_log = len(category_comparisons)
        print(
            f"Found {num_total_comparisons_in_log} comparison items in log for this category."
        )

        # 2. Calculate Judge Agreement Score & Decisiveness Score
        # Group by (sentence, model_a, model_b) to assess inter-judge agreement
        grouped_for_judge_agreement = defaultdict(list)
        for item in category_comparisons:
            key = (
                item.sentence,
                item.tested_entry_a.unique_id(),
                item.tested_entry_b.unique_id(),
            )
            grouped_for_judge_agreement[key].append(item)

        agreement_scores = []
        decisiveness_scores = []

        num_unique_eval_triplets = len(grouped_for_judge_agreement)
        if num_unique_eval_triplets == 0:
            category_metrics.append(
                {
                    "category": category_name,
                    "num_total_comparisons_in_log": num_total_comparisons_in_log,
                    "judge_agreement_score": 0,
                    "decisiveness_score": 0,
                    "bt_strength_spread": 0,
                    "bt_avg_ci_width": float("inf"),
                    "bt_significant_pairs_ratio": 0,
                    "bt_model_fitted": False,
                    "composite_signal_score": 0,
                    "notes": "No unique (sentence, m_a, m_b) evaluation triplets.",
                }
            )
            print(
                "No unique (sentence, m_a, m_b) evaluation triplets. Skipping judge agreement."
            )
            continue

        for eval_triplet_key, items_for_triplet in grouped_for_judge_agreement.items():
            n_a_wins = sum(1 for item in items_for_triplet if item.a_success)
            n_b_wins = sum(1 for item in items_for_triplet if item.b_success)
            n_identical = sum(1 for item in items_for_triplet if item.identical)
            n_total_judges = len(items_for_triplet)

            if n_total_judges > 0:
                # Judge Agreement: proportion of judges that agreed on the most common outcome
                agreement_scores.append(
                    max(n_a_wins, n_b_wins, n_identical) / n_total_judges
                )
                # Decisiveness: proportion of judgments that were NOT "identical"
                decisiveness_scores.append((n_a_wins + n_b_wins) / n_total_judges)

        avg_judge_agreement = np.mean(agreement_scores) if agreement_scores else 0
        avg_decisiveness = np.mean(decisiveness_scores) if decisiveness_scores else 0

        print(f"Avg. Judge Agreement: {avg_judge_agreement:.3f}")
        print(f"Avg. Decisiveness: {avg_decisiveness:.3f}")

        # 3. Fit a category-specific BT model
        bt_strength_spread = 0
        bt_avg_ci_width = float("inf")  # Higher is worse
        bt_significant_pairs_ratio = 0
        bt_model_fitted = False
        notes = ""

        # Need to create id_model_pairs specific to this category's comparisons
        # Wrap this in a try-except as produce_id_model_pairs might fail if no models
        temp_relevant_comparison_sets = [
            {
                "model_a": comp.tested_entry_a,
                "model_b": comp.tested_entry_b,
                # dummy 'comparisons' value, not used by produce_id_model_pairs
                "comparisons": [],
            }
            for comp in category_comparisons
        ]

        try:
            category_id_model_pairs = produce_id_model_pairs(
                temp_relevant_comparison_sets
            )
        except Exception as e:
            print(f"Could not produce id_model_pairs for {category_name}: {e}")
            category_id_model_pairs = []

        if (
            not category_id_model_pairs
            or len(category_id_model_pairs) < min_models_for_bt
        ):
            notes += (
                f"Not enough unique models ({len(category_id_model_pairs)}) for BT. "
            )
            print(notes)
        elif len(category_comparisons) < min_comparisons_for_bt:
            notes += f"Not enough comparisons ({len(category_comparisons)}) for BT. "
            print(notes)
        else:
            try:
                # Convert ComparisonItem to the format expected by DavidsonBT.from_comparisons
                bt_input_comparisons = []
                for item in category_comparisons:
                    id_a, id_b = -1, -1
                    for id_val, model_obj in category_id_model_pairs:
                        if model_obj == item.tested_entry_a:
                            id_a = id_val
                        if model_obj == item.tested_entry_b:
                            id_b = id_val

                    if id_a != -1 and id_b != -1:  # Both models found in id_map
                        if item.a_success:
                            bt_input_comparisons.append((id_a, id_b, "win1"))
                        elif item.b_success:
                            bt_input_comparisons.append((id_a, id_b, "win2"))
                        elif item.identical:
                            bt_input_comparisons.append((id_a, id_b, "tie"))

                if not bt_input_comparisons:
                    notes += "No valid comparisons for BT model after ID mapping. "
                    print(notes)
                else:
                    print(
                        f"Fitting BT model for {category_name} with {len(bt_input_comparisons)} comparisons and {len(category_id_model_pairs)} models."
                    )
                    category_bt_model = DavidsonBT.from_comparisons(
                        bt_input_comparisons, n_items=len(category_id_model_pairs)
                    )
                    bt_model_fitted = True

                    strengths = category_bt_model.get_strengths()
                    if strengths is not None and len(strengths) > 1:
                        bt_strength_spread = np.std(strengths)
                    else:
                        bt_strength_spread = 0

                    cis = category_bt_model.get_confidence_intervals(
                        conf_level=0.95
                    )  # Using 95% CI
                    ci_widths = [high - low for low, mid, high in cis]
                    if ci_widths:
                        bt_avg_ci_width = np.mean(ci_widths)

                    p_values_full = category_bt_model.pairwise_p_values_full()
                    num_significant_pairs = 0
                    num_total_model_pairs = 0
                    if p_values_full:
                        # n_models = len(category_id_model_pairs)
                        # num_total_model_pairs = n_models * (n_models - 1) / 2 # Unique pairs i < j
                        # This is already handled by pairwise_p_values_full which gives i!=j
                        num_total_model_pairs = len(p_values_full)

                        for (m1_id, m2_id), p_val in p_values_full.items():
                            if m1_id < m2_id:  # Count each pair once
                                if p_val < p_value_threshold:
                                    num_significant_pairs += 1

                        if num_total_model_pairs > 0:
                            # We want ratio of unique pairs i < j
                            actual_unique_pairs = len(
                                category_bt_model.pairwise_p_values()
                            )
                            if actual_unique_pairs > 0:
                                bt_significant_pairs_ratio = (
                                    num_significant_pairs / actual_unique_pairs
                                )
                            else:
                                bt_significant_pairs_ratio = 0

                    print(f"  BT Strength Spread: {bt_strength_spread:.4f}")
                    print(f"  BT Avg. CI Width: {bt_avg_ci_width:.4f}")
                    print(
                        f"  BT Significant Pairs Ratio ({p_value_threshold=}): {bt_significant_pairs_ratio:.3f}"
                    )

            except Exception as e:
                notes += f"Error fitting BT model for category {category_name}: {e}. "
                print(notes)
                # Ensure defaults are set if BT fitting fails mid-way
                bt_strength_spread = 0
                bt_avg_ci_width = float("inf")
                bt_significant_pairs_ratio = 0
                bt_model_fitted = False

        # Composite Score (example, can be tuned)
        # We want high agreement, high decisiveness, high spread, low CI width, high sig pairs
        # Normalize or rank-based combination might be better, but for simplicity:
        score_agreement = avg_judge_agreement
        score_decisiveness = avg_decisiveness
        score_spread = bt_strength_spread * 10  # Scale up spread as it's often small
        score_ci = 1 / (
            bt_avg_ci_width + 1e-6
        )  # Inverse, add epsilon to avoid div by zero
        score_sig_pairs = bt_significant_pairs_ratio

        # Weights can be adjusted based on what you prioritize
        # Here, giving more weight to BT model's ability to separate and judge consistency
        composite_signal_score = (
            (
                0.3 * score_agreement
                + 0.1 * score_decisiveness
                + 0.25 * score_spread
                + 0.15 * score_ci
                + 0.2 * score_sig_pairs
            )
            if bt_model_fitted
            else (0.6 * score_agreement + 0.4 * score_decisiveness)
        )  # Fallback if BT not fitted

        category_metrics.append(
            {
                "category": category_name,
                "num_total_comparisons_in_log": num_total_comparisons_in_log,
                "num_unique_eval_triplets": num_unique_eval_triplets,
                "judge_agreement_score": avg_judge_agreement,
                "decisiveness_score": avg_decisiveness,
                "bt_strength_spread": bt_strength_spread,
                "bt_avg_ci_width": bt_avg_ci_width,
                "bt_significant_pairs_ratio": bt_significant_pairs_ratio,
                "bt_model_fitted": bt_model_fitted,
                "composite_signal_score": composite_signal_score,
                "notes": notes.strip(),
            }
        )

    # Sort categories by composite score, descending (higher is better)
    return sorted(
        category_metrics, key=lambda x: x["composite_signal_score"], reverse=True
    )


def print_category_signal_analysis(analysis_results):
    print("\n\n--- Sentence Category Signal Strength Analysis ---")
    print("Sorted by Composite Signal Score (Higher is Better)")
    print("=" * 120)
    header = (
        f"{'Category':<20} | {'Comp. Score':>12} | {'Judge Agree.':>12} | "
        f"{'Decisive.':>10} | {'BT Spread':>10} | {'BT CI Width':>12} | "
        f"{'BT Sig.Pair%':>12} | {'BT Fit':<6} | {'Notes'}"
    )
    print(header)
    print("-" * 120)
    for r in analysis_results:
        row = (
            f"{r['category']:<20} | {r['composite_signal_score']:>12.3f} | "
            f"{r['judge_agreement_score']:>12.3f} | {r['decisiveness_score']:>10.3f} | "
            f"{r['bt_strength_spread']:>10.4f} | "
            f"{r['bt_avg_ci_width']:>12.4f} | {r['bt_significant_pairs_ratio']:>12.3f} | "
            f"{str(r['bt_model_fitted']):<6} | {r['notes']}"
        )
        print(row)
    print("=" * 120)
    print("\nField Explanations:")
    print(
        "  Comp. Score: Composite score indicating overall signal strength. Higher is better."
    )
    print(
        "  Judge Agree.: Average proportion of judges agreeing on the most common outcome for a given (sentence, A, B) triplet."
    )
    print("  Decisive.: Average proportion of non-'Identical' judgments for a triplet.")
    print(
        "  BT Spread: Standard deviation of model strengths from category-specific BT model. Higher = better separation."
    )
    print(
        "  BT CI Width: Average width of 95% confidence intervals for model strengths. Lower = more certainty."
    )
    print(
        "  BT Sig.Pair%: Ratio of significantly different model pairs (p < threshold) from category-specific BT."
    )
    print("  BT Fit: Whether a BT model was successfully fitted for the category.")


def analyze_judge_performance(
    all_comparison_items: list[ComparisonItem],
    all_tested_models: list[TestedEntry],
    judge_names: list[str],  # This will now correctly be a list of strings
):
    """
    Analyzes the performance of different judges.
    (docstring as before)
    """
    judge_analysis_results = {}

    # --- 1. Per-Judge Basic Statistics (Agreement with Majority, Decisiveness) ---
    # ... (this part should be mostly fine, but ensure judge_name in judge_stats is a string)
    comparison_instance_votes = defaultdict(list)
    for item in all_comparison_items:
        instance_key = (
            item.sentence,
            item.tested_entry_a.unique_id(),
            item.tested_entry_b.unique_id(),
        )
        vote = None
        if item.a_success:
            vote = "win_a"
        elif item.b_success:
            vote = "win_b"
        elif item.identical:
            vote = "tie"

        if vote:
            comparison_instance_votes[instance_key].append(
                {
                    "judge": item.evaluating_model,  # This is already a string
                    "vote": vote,
                }
            )

    # Initialize judge_stats with string keys
    judge_stats = {
        name_str: {  # name_str is now guaranteed to be a string
            "agreements_with_majority": 0,
            "total_decisions_in_majority_contests": 0,
            "decisions_made": 0,
            "ties_called": 0,
            "a_wins_called": 0,
            "b_wins_called": 0,
            "agreement_rate_with_majority": 0.0,
            "tie_rate": 0.0,
            "bt_model_fitted": False,
            "bt_score_pearson_corr_overall": None,
            "bt_rank_spearman_corr_overall": None,
        }
        for name_str in judge_names  # judge_names is now list[str]
    }
    # ... (rest of section 1 logic for populating judge_stats remains the same,
    #      as v_info["judge"] is already a string from item.evaluating_model) ...
    for instance_key, votes_for_instance in comparison_instance_votes.items():
        if not votes_for_instance:
            continue
        vote_counts = defaultdict(int)
        for v_info in votes_for_instance:
            vote_counts[v_info["vote"]] += 1
        if not vote_counts:
            continue
        majority_vote = max(vote_counts, key=vote_counts.get)
        for v_info in votes_for_instance:
            judge_name_str = v_info["judge"]  # This is a string
            if judge_name_str not in judge_stats:  # Check against string keys
                print(
                    f"Warning: Judge '{judge_name_str}' from comparison item not in provided judge_names list. Skipping."
                )
                continue
            judge_stats[judge_name_str]["total_decisions_in_majority_contests"] += 1
            if v_info["vote"] == majority_vote:
                judge_stats[judge_name_str]["agreements_with_majority"] += 1
            judge_stats[judge_name_str]["decisions_made"] += 1
            if v_info["vote"] == "win_a":
                judge_stats[judge_name_str]["a_wins_called"] += 1
            elif v_info["vote"] == "win_b":
                judge_stats[judge_name_str]["b_wins_called"] += 1
            elif v_info["vote"] == "tie":
                judge_stats[judge_name_str]["ties_called"] += 1

    for name_str in judge_names:
        if judge_stats[name_str]["total_decisions_in_majority_contests"] > 0:
            judge_stats[name_str]["agreement_rate_with_majority"] = (
                judge_stats[name_str]["agreements_with_majority"]
                / judge_stats[name_str]["total_decisions_in_majority_contests"]
            )
        if judge_stats[name_str]["decisions_made"] > 0:
            judge_stats[name_str]["tie_rate"] = (
                judge_stats[name_str]["ties_called"]
                / judge_stats[name_str]["decisions_made"]
            )

    judge_analysis_results["stats_by_judge"] = judge_stats

    # --- 2. Bradley-Terry Model Comparison ---
    # `all_tested_models` is the list of TestedEntry objects for the current evaluation.
    # Their index in this list will serve as their integer ID for the BT model.

    # Create a mapping from a model's unique_id string to its integer ID (index)
    unique_id_to_int_id_map = {
        model_obj.unique_id(): i for i, model_obj in enumerate(all_tested_models)
    }

    # This list is used by produce_sane_data_from_model. It needs (int_id, TestedEntry_object).
    all_id_model_pairs_for_bt = [
        (i, model_obj) for i, model_obj in enumerate(all_tested_models)
    ]

    # Fit overall BT model (using all judges' decisions for the current language)
    overall_bt_comparisons_input = []
    for (
        item
    ) in all_comparison_items:  # `all_comparison_items` are for the current language
        id_a = unique_id_to_int_id_map.get(item.tested_entry_a.unique_id())
        id_b = unique_id_to_int_id_map.get(item.tested_entry_b.unique_id())

        if id_a is None or id_b is None:
            missing_a_id = item.tested_entry_a.unique_id() if id_a is None else ""
            missing_b_id = item.tested_entry_b.unique_id() if id_b is None else ""
            print(
                f"Warning: Model(s) '{missing_a_id}{missing_b_id}' from comparison item "
                f"not in all_tested_models map during overall BT fitting. Skipping this item."
            )
            continue

        if item.a_success:
            overall_bt_comparisons_input.append((id_a, id_b, "win1"))
        elif item.b_success:
            overall_bt_comparisons_input.append((id_a, id_b, "win2"))
        elif item.identical:
            overall_bt_comparisons_input.append((id_a, id_b, "tie"))

    overall_bt_model = None
    overall_strengths_map = {}
    overall_ranks_map = {}

    if len(overall_bt_comparisons_input) >= 10 and len(all_tested_models) >= 2:
        try:
            overall_bt_model = DavidsonBT.from_comparisons(
                overall_bt_comparisons_input, n_items=len(all_tested_models)
            )
            raw_overall_strengths = overall_bt_model.get_strengths()
            for i, strength in enumerate(raw_overall_strengths):  # i is the int_id
                overall_strengths_map[i] = strength

            sorted_overall_by_strength = sorted(
                overall_strengths_map.items(),
                key=lambda x_item: x_item[1],
                reverse=True,
            )
            for rank, (model_id, strength) in enumerate(sorted_overall_by_strength):
                overall_ranks_map[model_id] = rank

            # THE CRITICAL FIX IS HERE: all_id_model_pairs_for_bt is now correctly (int, TestedEntry)
            judge_analysis_results["overall_bt_model_summary"] = (
                produce_sane_data_from_model(
                    all_id_model_pairs_for_bt, overall_bt_model
                )
            )
        except Exception as e:
            print(f"Could not fit overall BT model for judge analysis: {e}")
            overall_bt_model = None
    else:
        print(
            f"Not enough data for overall BT model in judge analysis. Comparisons: {len(overall_bt_comparisons_input)}, Models: {len(all_tested_models)}"
        )

    # Fit BT model for each judge and compare to overall
    for judge_name_str in judge_names:  # judge_names is now list[str]
        if not overall_bt_model:
            judge_stats[judge_name_str]["bt_model_fitted"] = False  # Use judge_name_str
            continue

        judge_specific_comparisons_input = []
        for item in all_comparison_items:
            if item.evaluating_model == judge_name_str:  # Compare string with string
                id_a = unique_id_to_int_id_map.get(item.tested_entry_a.unique_id())
                id_b = unique_id_to_int_id_map.get(item.tested_entry_b.unique_id())

                if id_a is None or id_b is None:
                    # Log if necessary, then skip
                    continue

                if item.a_success:
                    judge_specific_comparisons_input.append((id_a, id_b, "win1"))
                elif item.b_success:
                    judge_specific_comparisons_input.append((id_a, id_b, "win2"))
                elif item.identical:
                    judge_specific_comparisons_input.append((id_a, id_b, "tie"))

        if len(judge_specific_comparisons_input) < 10 or len(all_tested_models) < 2:
            judge_stats[judge_name_str]["bt_model_fitted"] = False  # Use judge_name_str
            print(
                f"Judge {judge_name_str} has insufficient data for BT model. Comparisons: {len(judge_specific_comparisons_input)}, Models: {len(all_tested_models)}"
            )
            continue

        try:
            judge_bt_model = DavidsonBT.from_comparisons(
                judge_specific_comparisons_input, n_items=len(all_tested_models)
            )
            judge_stats[judge_name_str]["bt_model_fitted"] = True  # Use judge_name_str

            judge_strengths_map = {
                i: strength for i, strength in enumerate(judge_bt_model.get_strengths())
            }
            sorted_judge_by_strength = sorted(
                judge_strengths_map.items(), key=lambda x_item: x_item[1], reverse=True
            )
            judge_ranks_map = {
                model_id: rank
                for rank, (model_id, strength) in enumerate(sorted_judge_by_strength)
            }

            common_model_ids = sorted(
                list(
                    set(overall_strengths_map.keys()) & set(judge_strengths_map.keys())
                )
            )

            if len(common_model_ids) > 1:
                overall_s_values = [
                    overall_strengths_map[mid] for mid in common_model_ids
                ]
                judge_s_values = [judge_strengths_map[mid] for mid in common_model_ids]
                overall_r_values = [overall_ranks_map[mid] for mid in common_model_ids]
                judge_r_values = [judge_ranks_map[mid] for mid in common_model_ids]

                if len(set(overall_s_values)) > 1 and len(set(judge_s_values)) > 1:
                    pearson_corr, _ = pearsonr(overall_s_values, judge_s_values)
                    judge_stats[judge_name_str][
                        "bt_score_pearson_corr_overall"
                    ] = pearson_corr

                if len(set(overall_r_values)) > 1 and len(set(judge_r_values)) > 1:
                    spearman_corr, _ = spearmanr(overall_r_values, judge_r_values)
                    judge_stats[judge_name_str][
                        "bt_rank_spearman_corr_overall"
                    ] = spearman_corr

        except Exception as e:
            print(
                f"Error fitting BT model for judge {judge_name_str}: {e}"
            )  # Use judge_name_str
            judge_stats[judge_name_str]["bt_model_fitted"] = False  # Use judge_name_str

    return judge_analysis_results


def print_judge_analysis(analysis_results):
    print("\n\n--- Judge Performance Analysis ---")
    stats_by_judge = analysis_results.get("stats_by_judge")
    if not stats_by_judge:
        print("No judge statistics found in analysis results.")
        return

    print("=" * 130)  # Adjusted width
    header = (
        f"{'Judge':<40} | {'Agree %':>8} | {'Decisions':>9} | {'Tie %':>6} | "
        f"{'Score Corr':>11} | {'Rank Corr':>10} | {'BT Fit':<6}"
    )
    print(header)
    print("-" * 130)

    sorted_judges = sorted(
        stats_by_judge.items(),
        key=lambda item: item[1].get("agreement_rate_with_majority", 0),
        reverse=True,
    )

    for judge_name, data in sorted_judges:
        agree_pct_str = f"{data.get('agreement_rate_with_majority', 0) * 100:.2f}%"
        decisions_str = str(data.get("decisions_made", 0))
        tie_pct_str = f"{data.get('tie_rate', 0) * 100:.2f}%"

        score_corr = data.get("bt_score_pearson_corr_overall")
        score_corr_str = f"{score_corr:.3f}" if score_corr is not None else "N/A"

        rank_corr = data.get("bt_rank_spearman_corr_overall")
        rank_corr_str = f"{rank_corr:.3f}" if rank_corr is not None else "N/A"

        bt_fitted_str = str(data.get("bt_model_fitted", False))

        row = (
            f"{judge_name:<40} | {agree_pct_str:>8} | {decisions_str:>9} | {tie_pct_str:>6} | "
            f"{score_corr_str:>11} | {rank_corr_str:>10} | {bt_fitted_str:<6}"
        )
        print(row)
    print("=" * 130)
    print("\nField Explanations:")
    print(
        "  Agree %: Judge agreement with majority vote on a specific (sentence, A, B) comparison."
    )
    print("  Decisions: Total comparisons evaluated by this judge.")
    print("  Tie %: Percentage of times judge declared a tie.")
    print(
        "  Score Corr: Pearson correlation of judge's BT strengths with overall BT strengths."
    )
    print(
        "  Rank Corr: Spearman correlation of judge's BT ranks with overall BT ranks."
    )
    print(
        "  BT Fit: If a Bradley-Terry model was fitted using only this judge's decisions."
    )
    print(
        "  N/A or False for correlations/BT Fit often means insufficient data for that judge."
    )

    if "overall_bt_model_summary" in analysis_results:
        print("\n--- Overall BT Model (All Judges) Summary ---")
        # Assuming print_model_rankings is available and works with the summary format
        print_model_rankings(
            analysis_results["overall_bt_model_summary"],
            language_name="Overall (All Judges - For Judge Analysis)",
        )
