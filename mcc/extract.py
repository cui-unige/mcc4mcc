#! /usr/bin/env python3

"""
Extract data from the Model Checking Contest results,
generate exact choice algorithms,
and learn from data for approximate algorithm.
"""

import argparse
import itertools
import json
import logging
import os
import statistics
import pickle
import pandas
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import tree
from ml_algorithm import init_algorithms
from global_variables import GlobalVariales
from processing import read_characteristics, read_results, set_techniques
from processing import rename_tools, sort_data, analyze_known
from score import best_time_of, max_score


def translate(what):
    """Translate values into numbers for machine learning algorithms."""
    if what is None:
        return 0
    if isinstance(what, (bool, str)):
        if what not in translate.ITEMS:
            translate.ITEMS[what] = translate.NEXT_ID + 1
            translate.NEXT_ID += 1
        return translate.ITEMS[what]
    else:
        return what


translate.ITEMS = {}
translate.NEXT_ID = 10


def translate_back(what):
    """Translate numbers into values from machine learning algorithms."""
    for wkey, wvalue in translate.ITEMS.items():
        if wvalue == what:
            return wkey
    return None


def powerset(iterable):
    """
    Computes the powerset of an iterable.
    See https://docs.python.org/2/library/itertools.html.
    """
    as_list = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(as_list, r) for r in range(len(as_list) + 1)
    )


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(
        description="Data extractor for the model checker collection"
    )
    PARSER.add_argument(
        "--results",
        help="results of the model checking contest",
        type=str,
        dest="results",
        default=os.getcwd() + "/results.csv",
    )
    PARSER.add_argument(
        "--characteristics",
        help="model characteristics from the Petri net repository",
        type=str,
        dest="characteristics",
        default=os.getcwd() + "/characteristics.csv",
    )
    PARSER.add_argument(
        "--known",
        help="data known from models",
        type=str,
        dest="known",
        default=os.getcwd() + "/known.json",
    )
    PARSER.add_argument(
        "--learned",
        help="data learned from models",
        type=str,
        dest="learned",
        default=os.getcwd() + "/learned.json",
    )
    PARSER.add_argument(
        "--iterations",
        help="Number of iterations",
        type=int,
        dest="iterations",
        default=0,
    )
    PARSER.add_argument(
        "--distance",
        help="Allowed distance from the best tool (in percent)",
        type=float,
        dest="distance",
    )
    PARSER.add_argument(
        "--duplicates",
        help="Allow duplicate entries",
        type=bool,
        dest="duplicates",
        default=False,
    )
    PARSER.add_argument(
        "--score",
        help="Compute score in the Model Checking Contest",
        type=bool,
        dest="mcc_score",
        default=True,
    )
    PARSER.add_argument(
        "--useless",
        help="Compute useless characteristics",
        type=bool,
        dest="useless",
        default=False,
    )
    PARSER.add_argument(
        "--output-dt",
        help="Output the graph of trained decision tree.",
        type=bool,
        dest="output_dt",
        default=False,
    )
    ARGUMENTS = PARSER.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    GV = GlobalVariales()
    GV.ALGORITHMS = init_algorithms(ARGUMENTS)

    logging.info(
        f"Reading model characteristics from '{ARGUMENTS.characteristics}'.")
    read_characteristics(ARGUMENTS, GV)

    logging.info(f"Reading mcc results from '{ARGUMENTS.results}'.")
    read_results(ARGUMENTS, GV)

    logging.info(f"Setting all techniques to Boolean values.")
    set_techniques(GV)

    logging.info(f"Renaming tools.")
    rename_tools(GV)

    GV.SIZE = len(GV.RESULTS)

    logging.info(f"Sorting data.")
    GV.SIZE = sort_data(GV)
    GV.DISTANCE = ARGUMENTS.distance

    logging.info(f"Analyzing known data.")
    analyze_known(GV)

    def mcc_score(alg_or_tool, g_v, to_drop=None):
        """
        Computes a score using the rules from the MCC.
        """
        score = {}
        for examination, models in g_v.KNOWN.items():
            score[examination] = 0
            for model, instances in models.items():
                if alg_or_tool in g_v.TOOLS:
                    tool = alg_or_tool
                else:
                    test = {}
                    test["Examination"] = translate(examination)
                    for key, value in g_v.CHARACTERISTICS[model].items():
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
        with tqdm(total=len(g_v.TOOLS)) as counter:
            for tool in GV.TOOLS:
                g_v.SCORES[tool] = mcc_score(tool, g_v)
                counter.update(1)

    if ARGUMENTS.mcc_score:
        logging.info(f"Computing scores.")
        compute_scores(GV)

    translate.ITEMS = {
        False: -1,
        None: 0,
        True: 1,
    }

    def analyze_learned(g_v):
        """
        Analyzes learned data.
        """
        with tqdm(total=len(g_v.RESULTS)) as counter:
            for _, entry in g_v.RESULTS.items():
                if entry["Year"] == g_v.TOOL_YEAR[entry["Tool"]] \
                        and "Selected" in entry \
                        and entry["Selected"]:
                    characteristics = {}
                    for key, value in entry.items():
                        if key not in g_v.REMOVE \
                                and key not in GV.TECHNIQUES:
                            characteristics[key] = translate(value)
                    g_v.LEARNED.append(characteristics)
                counter.update(1)
        logging.info(f"Select {len (g_v.LEARNED)} best entries.")
        # Convert this dict into dataframe:
        dataframe = pandas.DataFrame(g_v.LEARNED)
        # Remove duplicate entries if required:
        if not ARGUMENTS.duplicates:
            dataframe = dataframe.drop_duplicates(keep="first")
        logging.info(f"Using {dataframe.shape [0]} non duplicate entries.")
        # Compute efficiency for each algorithm:
        for name, algorithm in GV.ALGORITHMS.items():
            subresults = []
            logging.info(f"Learning using algorithm: '{name}'.")
            alg_results = {
                "algorithm": name,
            }
            if ARGUMENTS.iterations > 0:
                for _ in tqdm(range(ARGUMENTS.iterations)):
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
            if ARGUMENTS.mcc_score:
                g_v.SCORES[name] = mcc_score(algorithm, g_v)
                for key, value in g_v.SCORES[name].items():
                    alg_results[key] = value
                total = g_v.SCORES[name]["Total"]
                logging.info(f"  Score   : {total}")
            g_v.ALGORITHMS_RESULTS.append(alg_results)
            with open(f"learned.{name}.p", "wb") as output:
                pickle.dump(algorithm, output)
        with open("learned.json", "w") as output:
            json.dump({
                "algorithms": g_v.ALGORITHMS_RESULTS,
                "translation": translate.ITEMS,
            }, output)
        if ARGUMENTS.mcc_score:
            logging.info(f"Maximum score is {max_score(g_v)}.")
            srt = []
            for name, score in g_v.SCORES.items():
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
        if ARGUMENTS.output_dt:
            if "decision-tree" in g_v.ALGORITHMS:
                tree.export_graphviz(
                    g_v.ALGORITHMS["decision-tree"](True).fit(
                        dataframe.drop("Tool", 1), dataframe["Tool"]
                    ),
                    feature_names=dataframe.drop("Tool", 1).columns,
                    filled=True, rounded=True,
                    special_characters=True
                )

    logging.info(f"Analyzing learned data.")
    analyze_learned(GV)

    print("ça va planter")

    def analyze_useless(g_v):
        """
        Analyzes useless characteristics.
        """
        # Build the dataframe:
        learned = []
        with tqdm(total=len(g_v.RESULTS)) as counter:
            for _, entry in g_v.RESULTS.items():
                if entry["Year"] == g_v.TOOL_YEAR[entry["Tool"]] \
                        and "Selected" in entry \
                        and entry["Selected"]:
                    characteristics = {}
                    for key, value in entry.items():
                        if key not in g_v.REMOVE \
                                and key not in g_v.TECHNIQUES:
                            characteristics[key] = translate(value)
                    learned.append(characteristics)
                counter.update(1)
        # Convert this dict into dataframe:
        dataframe = pandas.DataFrame(learned)
        # Remove duplicate entries if required:
        if not ARGUMENTS.duplicates:
            dataframe = dataframe.drop_duplicates(keep="first")
        logging.info(f"Using {dataframe.shape [0]} non duplicate entries.")
        results = {}
        # For each algorithm, try to drop each characteristic,
        # and compare the score with the same with all characteristics:
        for name, falgorithm in g_v.ALGORITHMS.items():
            useless = {}
            for characteristic in g_v.TO_DROP:
                useless[characteristic] = True
            logging.info(f"Analyzing characteristics in algorithm {name}.")
            results[name] = {}
            with tqdm(total=len(g_v.TO_DROP)) as counter:
                for to_drop in g_v.TO_DROP:
                    algorithm = falgorithm(True)
                    algorithm.fit(
                        dataframe.drop("Tool", 1).drop(to_drop, 1),
                        dataframe["Tool"]
                    )
                    score = mcc_score(algorithm, g_v, to_drop)
                    # If the score has changed,
                    # the characteristic is not useless:
                    if score != g_v.SCORES[name]:
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
                            if score != g_v.SCORES[name]:
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

    if ARGUMENTS.useless and ARGUMENTS.mcc_score:
        logging.info(f"Analyzing useless characteristics.")
        analyze_useless(GV)
