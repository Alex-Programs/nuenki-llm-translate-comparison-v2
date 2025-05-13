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


def process_sentence(
    language, category, sentence, model_a, model_b, cache, compare_models, output_queue
):
    print("Checking", language, category, sentence)

    def get_translation_with_cache_check(model):
        cache_key = f"TRANSLATION Language:{language.value}|Model: {model.unique_id()}|Sentence category:{category}|Sentence md5:{md5hash(sentence)}"
        check = cache.get(cache_key)
        if check:
            return check
        else:
            print("Translating", cache_key)

            # this is to avoid getting ratelimited...
            sleep_t = choice(range(0, 4))
            time.sleep(sleep_t)

            start_t = None
            end_t = None
            translation = None

            i = 0
            while True:
                print("Translation attempt", i)
                i += 1
                try:
                    start_t = time.time()
                    translation = model.inference_source.translate(
                        TranslatableLanguage.English, language, sentence, model.temp
                    )
                    end_t = time.time()
                    break
                except Exception as e:
                    print("Err on translate", e, "with m", model)
                    time.sleep(i * choice(range(3, 12)))

            cache.set(cache_key, translation)
            log_latency(model.unique_id(), category, sentence, end_t - start_t)
            return translation

    model_a_translation = get_translation_with_cache_check(model_a)
    model_b_translation = get_translation_with_cache_check(model_b)

    for comparison_model, comparison_inference in compare_models:
        print("MMM", model_a, model_b)
        print("Comparison with ", comparison_model)

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
            print("Both refused!")
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
            print("Only A refused")
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
            print("Only B refused")
            continue

        # another special case: If they're literally the same, just write down Identical
        if model_a_translation.strip() == model_b_translation.strip():
            print("Identical-Skip")
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

        print(model_a_translation, "::", model_b_translation)

        coin_flip_key = f"COIN FLIP Language:{language.value}|Sentence cat:{category}|Sentence md5:{md5hash(sentence)}|{comparison_model}|{model_a.unique_id()}|{model_b.unique_id()}"
        swap_a_b = deterministic_coin_flip(coin_flip_key)

        if swap_a_b:
            print("SWAPPING ORDER!!")
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
        print("KEY", key)
        comparison_data = None

        if cache.get(key):
            comparison_data = cache.get(key)
            print("Cache hit")
        else:
            print("Inference for comparison")
            wait_t = choice(range(0, 6))
            time.sleep(wait_t)

            for i in range(3):
                try:
                    comparison_data = comparison_inference.infer(comparison_prompt, 0)
                    if comparison_data:
                        break
                except Exception as e:
                    print(
                        "Error during inference:",
                        e,
                        "with comp model",
                        comparison_model,
                    )
                    time.sleep(i * choice(range(5, 25)))

        if not comparison_data:
            continue

        print("Comparison data", comparison_data)
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
                print(
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

                print(
                    f"TRY REPLACE: LAST LINE UNSWAPPED AFTER: `{last_line_unswapped}`"
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
            print("Reports for multiple in response: " + str(comparison_data))
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
            print("Cannot get selection from: ", comparison_data)


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

    # iterative active learning loop using entropy

    MIN_ENTROPY = 1.05

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
