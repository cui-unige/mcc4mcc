"""
This module contains the scoring functions.
"""
import json
import logging
import statistics
import pickle
import pandas
from sklearn.model_selection import train_test_split
from sklearn import tree
from tqdm import tqdm
from processing import translate, translate_back, powerset


def best_time_of(sequence, seq_key):
    """
    Computes the time of the best in sequence, sorted by seq_key.
    """
    rbest = sorted(
        sequence,
        key=lambda e: e[seq_key]
    )
    if rbest:
        return rbest[0]["time"]
    return None


def max_score(g_v):
    """
    Computes the maximum score using the rules from the MCC.
    """
    score = 0
    for _, models in g_v.known.items():
        for _ in models.items():
            score += 16 + 2 + 2
    return int(score)


def mcc_score(alg_or_tool, g_v, to_drop=None):
    """
    Computes a score using the rules from the MCC.
    """
    score = {}
    for examination, models in g_v.known.items():
        score[examination] = 0
        for model, instances in models.items():
            if alg_or_tool in g_v.tools:
                tool = alg_or_tool
            else:
                test = {}
                test["Examination"] = translate(examination)
                for key, value in g_v.characteristics[model].items():
                    test[key] = translate(value)
                del test["Id"]
                del test["Parameterised"]
                dataframe = pandas.DataFrame([test])
                if to_drop is not None:
                    dataframe = dataframe.drop(to_drop, 1)
                tool = translate_back(alg_or_tool.predict(dataframe)[0])
            subscore = 0
            for instance, data in instances.items():
                if instance == "sorted":
                    continue
                for entry in data["sorted"]:
                    if entry["tool"] != tool:
                        continue
                    bestt = best_time_of(data["sorted"], "time")
                    bestm = best_time_of(data["sorted"], "memory")
                    subscore += 16 + \
                        (2 if entry["time"] == bestt else 0) + \
                        (2 if entry["memory"] == bestm else 0)
                    break
            score[examination] += subscore / (len(instances) - 1)
    full_score = 0
    for examination, value in score.items():
        full_score += value
        score[examination] = int(value)
    score["Total"] = int(full_score)
    return score


def compute_scores(g_v):
    """
    Computes the scores of all tools.
    """
    with tqdm(total=len(g_v.tools)) as counter:
        for tool in g_v.tools:
            g_v.scores[tool] = mcc_score(tool, g_v)
            counter.update(1)


def analyze_learned(arguments, g_v):
    """
    Analyzes learned data.
    """
    with tqdm(total=len(g_v.results)) as counter:
        for _, entry in g_v.results.items():
            if entry["Year"] == g_v.tool_year[entry["Tool"]] \
                    and "Selected" in entry \
                    and entry["Selected"]:
                characteristics = {}
                for key, value in entry.items():
                    if key not in g_v.remove \
                            and key not in g_v.techniques:
                        characteristics[key] = translate(value)
                g_v.learned.append(characteristics)
            counter.update(1)
    logging.info(f"Select {len (g_v.learned)} best entries.")
    # Convert this dict into dataframe:
    dataframe = pandas.DataFrame(g_v.learned)
    # Remove duplicate entries if required:
    if not arguments.duplicates:
        dataframe = dataframe.drop_duplicates(keep="first")
    logging.info(f"Using {dataframe.shape [0]} non duplicate entries.")
    # Compute efficiency for each algorithm:
    for name, algorithm in g_v.algorithms.items():
        subresults = []
        logging.info(f"Learning using algorithm: '{name}'.")
        alg_results = {
            "algorithm": name,
        }
        if arguments.iterations > 0:
            for _ in tqdm(range(arguments.iterations)):
                train, test = train_test_split(dataframe)
                training_x = train.drop("Tool", 1)
                training_y = train["Tool"]
                test_x = test.drop("Tool", 1)
                test_y = test["Tool"]
                # Apply algorithm:
                algorithm.fit(training_x, training_y)
                subresults.append(algorithm.score(test_x, test_y))
            alg_results["min"] = min(subresults)
            alg_results["max"] = max(subresults)
            alg_results["mean"] = statistics.mean(subresults)
            alg_results["median"] = statistics.median(subresults)
            logging.info(f"Algorithm: {name}")
            logging.info(f"  Min     : {min                (subresults)}")
            logging.info(f"  Max     : {max                (subresults)}")
            logging.info(f"  Mean    : {statistics.mean    (subresults)}")
            logging.info(f"  Median  : {statistics.median  (subresults)}")
        algorithm.fit(dataframe.drop("Tool", 1), dataframe["Tool"])
        if arguments.mcc_score:
            g_v.scores[name] = mcc_score(algorithm, g_v)
            for key, value in g_v.scores[name].items():
                alg_results[key] = value
            total = g_v.scores[name]["Total"]
            logging.info(f"  Score   : {total}")
        g_v.algorithms_results.append(alg_results)
        with open(f"learned.{name}.p", "wb") as output:
            pickle.dump(algorithm, output)
    with open("learned.json", "w") as output:
        json.dump({
            "algorithms": g_v.algorithms_results,
            "translation": translate.ITEMS,
        }, output)
    if arguments.mcc_score:
        logging.info(f"Maximum score is {max_score(g_v)}.")
        srt = []
        for name, score in g_v.scores.items():
            for examination, value in score.items():
                srt.append({
                    "name": name,
                    "examination": examination,
                    "score": value,
                })
        srt = sorted(srt, key=lambda e: (
            e["examination"], e["score"], e["name"]
        ), reverse=True)
        for element in srt:
            examination = element["examination"]
            score = element["score"]
            name = element["name"]
            logging.info(f"In {examination} : {score} for {name}.")
    if arguments.output_dt:
        if "decision-tree" in g_v.algorithms:
            tree.export_graphviz(
                g_v.algorithms["decision-tree"](True).fit(
                    dataframe.drop("Tool", 1), dataframe["Tool"]
                ),
                feature_names=dataframe.drop("Tool", 1).columns,
                filled=True, rounded=True,
                special_characters=True
            )


def analyze_useless(arguments, g_v):
    """
    Analyzes useless characteristics.
    """
    # Build the dataframe:
    learned = []
    with tqdm(total=len(g_v.results)) as counter:
        for _, entry in g_v.results.items():
            if entry["Year"] == g_v.tool_year[entry["Tool"]] \
                    and "Selected" in entry \
                    and entry["Selected"]:
                characteristics = {}
                for key, value in entry.items():
                    if key not in g_v.remove \
                            and key not in g_v.techniques:
                        characteristics[key] = translate(value)
                learned.append(characteristics)
            counter.update(1)
    # Convert this dict into dataframe:
    dataframe = pandas.DataFrame(learned)
    # Remove duplicate entries if required:
    if not arguments.duplicates:
        dataframe = dataframe.drop_duplicates(keep="first")
    logging.info(f"Using {dataframe.shape [0]} non duplicate entries.")
    results = {}
    # For each algorithm, try to drop each characteristic,
    # and compare the score with the same with all characteristics:
    for name, falgorithm in g_v.algorithms.items():
        useless = {}
        for characteristic in g_v.to_drop:
            useless[characteristic] = True
        logging.info(f"Analyzing characteristics in algorithm {name}.")
        results[name] = {}
        with tqdm(total=len(g_v.to_drop)) as counter:
            for to_drop in g_v.to_drop:
                algorithm = falgorithm(True)
                algorithm.fit(
                    dataframe.drop("Tool", 1).drop(to_drop, 1),
                    dataframe["Tool"]
                )
                score = mcc_score(algorithm, g_v, to_drop)
                # If the score has changed,
                # the characteristic is not useless:
                if score != g_v.scores[name]:
                    useless[to_drop] = False
                counter.update(1)
        # The set of potential useless characteristics is obtained:
        useless = set([x for x, y in useless.items() if y])
        # If empty, there is no need for further investigation:
        if useless is None:
            return
        logging.info(f"  Some characteristics in {useless} are useless.")
        all_related = []
        # Try to find which characteristics are truly useless,
        # and which ones are linked to others.
        # To do so, build tuples of n characteristics (n growing from 2),
        # and try to remove them from the model.
        for length in range(2, len(useless)):
            sets = [list(x) for x in powerset(useless) if len(x) == length]
            related = []
            with tqdm(total=len(sets)) as counter:
                for characteristics in sets:
                    # If a part of the tuple is already linked,
                    # there is no need for investigation,
                    # as the result will be linked:
                    is_interesting = True
                    for rel in all_related:
                        if rel < set(characteristics):
                            is_interesting = False
                            break
                    if is_interesting:
                        algorithm = falgorithm(True)
                        algorithm.fit(
                            dataframe.drop("Tool", 1)
                            .drop(characteristics, 1),
                            dataframe["Tool"]
                        )
                        score = mcc_score(algorithm, g_v, characteristics)
                        # If the score has changed, the characteristics
                        # are linked:
                        if score != g_v.scores[name]:
                            related.append(set(characteristics))
                    counter.update(1)
            if not related:
                break
            for rel in related:
                logging.info(f"  Characteristics {rel} are related.")
                all_related.append(rel)
        # Characteristics that are linked to none are truly useless:
        for rel in all_related:
            useless -= rel
        logging.info(f"  Characteristics {useless} are useless.")
