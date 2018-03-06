#! /usr/bin/env python3

"""
Extract data from the Model Checking Contest results,
generate exact choice algorithms,
and learn from data for approximate algorithm.
"""

import argparse
import csv
import itertools
import json
import logging
import os
import re
import statistics
import pickle
import pandas
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import tree
from ml_algorithm import init_algorithms
from global_variables import GlobalVariales


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


def value_of(what):
    """Convert a string, such as True, Yes, ... to its real value."""
    if what in ["TRUE", "True", "Yes", "OK"]:
        return True
    if what in ["FALSE", "False", "None"]:
        return False
    if what == "Unknown":
        return None
    try:
        return int(what)
    except ValueError:
        pass
    try:
        return float(what)
    except ValueError:
        pass
    return what


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
    GV = GlobalVariales(ALGORITHMS=init_algorithms(ARGUMENTS))

    def read_characteristics():
        """
        Reads the model characteristics.
        """
        with tqdm(total=sum(
            1 for line in open(ARGUMENTS.characteristics)) - 1
                 ) as counter:
            with open(ARGUMENTS.characteristics) as data:
                data.readline()  # skip the title line
                reader = csv.reader(data)
                for row in reader:
                    entry = {}
                    for i, characteristic in enumerate(GV.CHARACTERISTIC_KEYS):
                        entry[characteristic] = value_of(row[i])
                    entry["Place/Transition"] = True if re.search(
                        "PT", entry["Type"]) else False
                    entry["Colored"] = True if re.search(
                        "COLORED", entry["Type"]) else False
                    del entry["Type"]
                    del entry["Fixed size"]
                    del entry["Origin"]
                    del entry["Submitter"]
                    del entry["Year"]
                    GV.CHARACTERISTICS[entry["Id"]] = entry
                    counter.update(1)
    logging.info(
        f"Reading model characteristics from '{ARGUMENTS.characteristics}'.")
    read_characteristics()

    RESULTS = {}

    def read_results():
        """
        Reads the results of the model checking contest.
        """
        with tqdm(total=sum(1 for line in open(ARGUMENTS.results)) - 1) \
                as counter:
            with open(ARGUMENTS.results) as data:
                data.readline()  # skip the title line
                reader = csv.reader(data)
                for row in reader:
                    entry = {}
                    for i, result in enumerate(GV.RESULT_KEYS):
                        entry[result] = value_of(row[i])
                    if entry["Time OK"] \
                            and entry["Memory OK"] \
                            and entry["Status"] == "normal" \
                            and entry["Results"] not in ["DNC", "DNF", "CC"]:
                        RESULTS[entry["Id"]] = entry
                        for technique in re.findall(
                                r"([A-Z_]+)",
                                entry["Techniques"]
                        ):
                            GV.TECHNIQUES[technique] = True
                            entry[technique] = True
                        entry["Surprise"] = True if re.search(
                            r"^S_", entry["Instance"]) else False
                        if entry["Surprise"]:
                            entry["Instance"] = re.search(
                                r"^S_(.*)$", entry["Instance"]).group(1)
                        split = re.search(
                            r"([^-]+)\-([^-]+)\-([^-]+)$", entry["Instance"])
                        if split is None:
                            entry["Model Id"] = entry["Instance"]
                        else:
                            entry["Model Id"] = split.group(1)
                        if entry["Model Id"] in GV.CHARACTERISTICS:
                            model = GV.CHARACTERISTICS[entry["Model Id"]]
                            for key in model.keys():
                                if key != "Id":
                                    entry[key] = model[key]
                        del entry["Time OK"]
                        del entry["Memory OK"]
                        del entry["CPU Time"]
                        del entry["Cores"]
                        del entry["IO Time"]
                        del entry["Results"]
                        del entry["Status"]
                        del entry["Techniques"]
                    counter.update(1)
    logging.info(f"Reading mcc results from '{ARGUMENTS.results}'.")
    read_results()

    def set_techniques():
        """
        Sets techniques to Boolean values in results.
        """
        with tqdm(total=len(RESULTS)) as counter:
            for _, entry in RESULTS.items():
                for technique in GV.TECHNIQUES:
                    if technique not in entry:
                        entry[technique] = False
                counter.update(1)
    logging.info(f"Setting all techniques to Boolean values.")
    set_techniques()

    def rename_tools():
        """
        Rename tools that are duplicated.
        """
        with tqdm(total=len(RESULTS)) as counter:
            for _, entry in RESULTS.items():
                name = entry["Tool"]
                if name in GV.TOOLS_RENAME:
                    entry["Tool"] = GV.TOOLS_RENAME[name]
                counter.update(1)
    logging.info(f"Renaming tools.")
    rename_tools()

    SIZE = len(RESULTS)
    DATA = {}
    TOOL_YEAR = {}

    def sort_data():
        """
        Sorts data into tree of examination/model/instance/tool/year/entry.
        """
        size = SIZE
        with tqdm(total=len(RESULTS)) as counter:
            for _, entry in RESULTS.items():
                if entry["Tool"] not in GV.TOOLS:
                    GV.TOOLS[entry["Tool"]] = True
                if entry["Examination"] not in DATA:
                    DATA[entry["Examination"]] = {}
                examination = DATA[entry["Examination"]]
                if entry["Model Id"] not in examination:
                    examination[entry["Model Id"]] = {}
                model = examination[entry["Model Id"]]
                if entry["Instance"] not in model:
                    model[entry["Instance"]] = {}
                instance = model[entry["Instance"]]
                if entry["Tool"] not in instance:
                    instance[entry["Tool"]] = {}
                tool = instance[entry["Tool"]]
                if entry["Tool"] not in TOOL_YEAR:
                    TOOL_YEAR[entry["Tool"]] = 0
                if entry["Year"] > TOOL_YEAR[entry["Tool"]]:
                    TOOL_YEAR[entry["Tool"]] = entry["Year"]
                if entry["Year"] in tool:
                    size -= 1
                    if entry["Clock Time"] < tool[entry["Year"]]["Clock Time"]:
                        tool[entry["Year"]] = entry
                else:
                    tool[entry["Year"]] = entry
                counter.update(1)
        return size

    logging.info(f"Sorting data.")
    SIZE = sort_data()

    KNOWN = {}
    DISTANCE = ARGUMENTS.distance

    def analyze_known():
        """
        Analyzes known data.
        """
        with tqdm(total=SIZE) as counter:
            for examination, models in DATA.items():
                KNOWN[examination] = {}
                known_e = KNOWN[examination]
                for model, instances in models.items():
                    known_e[model] = {}
                    known_m = known_e[model]
                    subresults = {}
                    for instance, tools in instances.items():
                        known_m[instance] = {}
                        known_i = known_m[instance]
                        subsubresults = {}
                        for tool, years in tools.items():
                            if tool not in subresults:
                                subresults[tool] = {
                                    "count": 0,
                                    "time": 0,
                                    "memory": 0,
                                }
                            for year, entry in years.items():
                                if year == TOOL_YEAR[tool]:
                                    subsubresults[tool] = {
                                        "time": entry["Clock Time"],
                                        "memory": entry["Memory"],
                                    }
                                    subresults[tool]["count"] += 1
                                    subresults[tool]["time"] += \
                                        entry["Clock Time"]
                                    subresults[tool]["memory"] += \
                                        entry["Memory"]
                                counter.update(1)
                        srt = sorted(subsubresults.items(), key=lambda e: (
                            e[1]["time"], e[1]["memory"]))
                        # Select only the tools that are within a distance
                        # from the best:
                        known_i["sorted"] = [
                            {"tool": x[0],
                             "time": x[1]["time"],
                             "memory": x[1]["memory"]} for x in srt]
                    srt = sorted(
                        subresults.items(),
                        key=lambda e: (
                            -e[1]["count"],
                            e[1]["time"],
                            e[1]["memory"]
                        )
                    )
                    known_m["sorted"] = [
                        {"tool": x[0],
                         "count": x[1]["count"],
                         "time": x[1]["time"],
                         "memory": x[1]["memory"]} for x in srt]
                    # Select all tools that reach both the maximum count and
                    # the expected distance from the best:
                    if known_m["sorted"]:
                        best = known_m["sorted"][0]
                        for x_e in known_m["sorted"]:
                            if x_e["count"] == best["count"]:
                                for instance, tools in instances.items():
                                    tool = x_e["tool"]
                                    ratio = x_e["time"] / best["time"]
                                    if tool in tools \
                                            and TOOL_YEAR[tool] in tools[tool]\
                                            and (DISTANCE is None
                                                 or ratio <= (1+DISTANCE)):
                                        entry = tools[tool][TOOL_YEAR[tool]]
                                        entry["Selected"] = True
        with open("known.json", "w") as output:
            json.dump(KNOWN, output)

    logging.info(f"Analyzing known data.")
    analyze_known()

    MAX_SCORE = 16 + 2 + 2

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

    def max_score():
        """
        Computes the maximum score using the rules from the MCC.
        """
        score = 0
        for _, models in KNOWN.items():
            for _ in models.items():
                score += 16 + 2 + 2
        return int(score)

    def mcc_score(alg_or_tool, to_drop=None):
        """
        Computes a score using the rules from the MCC.
        """
        score = {}
        for examination, models in KNOWN.items():
            score[examination] = 0
            for model, instances in models.items():
                if alg_or_tool in GV.TOOLS:
                    tool = alg_or_tool
                else:
                    test = {}
                    test["Examination"] = translate(examination)
                    for key, value in GV.CHARACTERISTICS[model].items():
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

    SCORES = {}

    def compute_scores():
        """
        Computes the scores of all tools.
        """
        with tqdm(total=len(GV.TOOLS)) as counter:
            for tool in GV.TOOLS:
                SCORES[tool] = mcc_score(tool)
                counter.update(1)

    if ARGUMENTS.mcc_score:
        logging.info(f"Computing scores.")
        compute_scores()

    LEARNED = []
    ALGORITHMS_RESULTS = []
    translate.ITEMS = {
        False: -1,
        None: 0,
        True: 1,
    }
    REMOVE = [
        "Id", "Model Id", "Instance", "Year",
        "Memory", "Clock Time",
        "Parameterised", "Selected", "Surprise"
    ]

    def analyze_learned():
        """
        Analyzes learned data.
        """
        with tqdm(total=len(RESULTS)) as counter:
            for _, entry in RESULTS.items():
                if entry["Year"] == TOOL_YEAR[entry["Tool"]] \
                        and "Selected" in entry \
                        and entry["Selected"]:
                    characteristics = {}
                    for key, value in entry.items():
                        if key not in REMOVE \
                                and key not in GV.TECHNIQUES:
                            characteristics[key] = translate(value)
                    LEARNED.append(characteristics)
                counter.update(1)
        logging.info(f"Select {len (LEARNED)} best entries.")
        # Convert this dict into dataframe:
        dataframe = pandas.DataFrame(LEARNED)
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
                SCORES[name] = mcc_score(algorithm)
                for key, value in SCORES[name].items():
                    alg_results[key] = value
                total = SCORES[name]["Total"]
                logging.info(f"  Score   : {total}")
            ALGORITHMS_RESULTS.append(alg_results)
            with open(f"learned.{name}.p", "wb") as output:
                pickle.dump(algorithm, output)
        with open("learned.json", "w") as output:
            json.dump({
                "algorithms": ALGORITHMS_RESULTS,
                "translation": translate.ITEMS,
            }, output)
        if ARGUMENTS.mcc_score:
            logging.info(f"Maximum score is {max_score()}.")
            srt = []
            for name, score in SCORES.items():
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
            if "decision-tree" in GV.ALGORITHMS:
                tree.export_graphviz(
                    GV.ALGORITHMS["decision-tree"](True).fit(
                        dataframe.drop("Tool", 1), dataframe["Tool"]
                    ),
                    feature_names=dataframe.drop("Tool", 1).columns,
                    filled=True, rounded=True,
                    special_characters=True
                )

    logging.info(f"Analyzing learned data.")
    analyze_learned()

    def analyze_useless():
        """
        Analyzes useless characteristics.
        """
        # Build the dataframe:
        learned = []
        with tqdm(total=len(RESULTS)) as counter:
            for _, entry in RESULTS.items():
                if entry["Year"] == TOOL_YEAR[entry["Tool"]] \
                        and "Selected" in entry \
                        and entry["Selected"]:
                    characteristics = {}
                    for key, value in entry.items():
                        if key not in REMOVE \
                                and key not in GV.TECHNIQUES:
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
        for name, falgorithm in GV.ALGORITHMS.items():
            useless = {}
            for characteristic in GV.TO_DROP:
                useless[characteristic] = True
            logging.info(f"Analyzing characteristics in algorithm {name}.")
            results[name] = {}
            with tqdm(total=len(GV.TO_DROP)) as counter:
                for to_drop in GV.TO_DROP:
                    algorithm = falgorithm(True)
                    algorithm.fit(
                        dataframe.drop("Tool", 1).drop(to_drop, 1),
                        dataframe["Tool"]
                    )
                    score = mcc_score(algorithm, to_drop)
                    # If the score has changed,
                    # the characteristic is not useless:
                    if score != SCORES[name]:
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
                            score = mcc_score(algorithm, characteristics)
                            # If the score has changed, the characteristics
                            # are linked:
                            if score != SCORES[name]:
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
        analyze_useless()
