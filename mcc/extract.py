#! /usr/bin/env python3

"""
Extract data from the Model Checking Contest results,
generate exact choice algorithms,
and learn from data for approximate algorithm.
"""

import argparse
import csv
import json
import logging
import os
import re
import statistics
import pickle
import pandas
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

CHARACTERISTIC_KEYS = [
    "Id",
    "Type",
    "Fixed size",
    "Parameterised",
    "Connected",
    "Conservative",
    "Deadlock",
    "Extended Free Choice",
    "Live",
    "Loop Free",
    "Marked Graph",
    "Nested Units",
    "Ordinary",
    "Quasi Live",
    "Reversible",
    "Safe",
    "Simple Free Choice",
    "Sink Place",
    "Sink Transition",
    "Source Place",
    "Source Transition",
    "State Machine",
    "Strongly Connected",
    "Sub-Conservative",
    "Origin",
    "Submitter",
    "Year",
]
RESULT_KEYS = [
    "Year",
    "Tool",
    "Instance",
    "Examination",
    "Cores",
    "Time OK",
    "Memory OK",
    "Results",
    "Techniques",
    "Memory",
    "CPU Time",
    "Clock Time",
    "IO Time",
    "Status",
    "Id",
]


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


def knn_distance(lhs, rhs, bound=2):
    """
    Fix the distance for knn, by taking into account that numbers greater
    than 1 represent enumerations.
    """
    # we suppose x and y have the same shape and are numpy array.
    # we create a mask for each vector. Values are true when it's
    # between [-1,+1]. The "*" operator is an "and"
    # We "and" the two mask and  change the values where the mask is True.
    lhs_mask = (lhs >= -1) * (lhs <= 1)
    rhs_mask = (rhs >= -1) * (rhs <= 1)
    diff = abs(lhs - rhs)
    diff[~(lhs_mask * rhs_mask)] = bound
    return diff.sum()


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
        default=10,
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
    ARGUMENTS = PARSER.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    ALGORITHMS = {}

    # Classificator parts:
    # Do not include these algorithms with duplicates,
    # as they are very slow.
    if not ARGUMENTS.duplicates:
        ALGORITHMS["knn"] = lambda _: KNeighborsClassifier(
            n_neighbors=10,
            weights="distance",
            metric=knn_distance,
        )
        ALGORITHMS["bagging-knn"] = lambda _: BaggingClassifier(
            KNeighborsClassifier(
                n_neighbors=10,
                weights="distance",
                metric=knn_distance
            ),
            max_samples=0.5,
            max_features=1,
            n_estimators=10,
        )

    ALGORITHMS["naive-bayes"] = lambda _: GaussianNB()

    ALGORITHMS["svm"] = lambda _: SVC()

    ALGORITHMS["ada-boost"] = lambda _: AdaBoostClassifier()

    ALGORITHMS["linear-svm"] = lambda _: LinearSVC()

    ALGORITHMS["decision-tree"] = lambda _: DecisionTreeClassifier()

    ALGORITHMS["random-forest"] = lambda _: RandomForestClassifier(
        n_estimators=20,
        max_features=None,
    )

    ALGORITHMS["neural-network"] = lambda _: MLPClassifier(
        solver="lbfgs",
    )

    # Custom data
    ALGORITHMS["my-algo"] = lambda _: MyAlgo()

    # Regressor part
    ALGORITHMS["decision-tree-regression"] = lambda _: DecisionTreeRegressor()

    ALGORITHMS["knn-regression"] = lambda _: KNeighborsRegressor(
        weights='distance', n_neighbors=30
    )

    ALGORITHMS["random-forest-regression"] = lambda _: RandomForestRegressor()

    TECHNIQUES = {}
    CHARACTERISTICS = {}

    logging.info(
        f"Reading model characteristics from '{ARGUMENTS.characteristics}'.")
    with tqdm(total=sum(
        1 for line in open(ARGUMENTS.characteristics)) - 1
             ) as counter:
        with open(ARGUMENTS.characteristics) as data:
            data.readline()  # skip the title line
            READER = csv.reader(data)
            for row in READER:
                entry = {}
                for i, characteristic in enumerate(CHARACTERISTIC_KEYS):
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
                CHARACTERISTICS[entry["Id"]] = entry
                counter.update(1)

    RESULTS = {}
    logging.info(f"Reading mcc results from '{ARGUMENTS.results}'.")
    with tqdm(total=sum(1 for line in open(ARGUMENTS.results)) - 1) as counter:
        with open(ARGUMENTS.results) as data:
            data.readline()  # skip the title line
            READER = csv.reader(data)
            for row in READER:
                entry = {}
                for i, result in enumerate(RESULT_KEYS):
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
                        TECHNIQUES[technique] = True
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
                    if entry["Model Id"] in CHARACTERISTICS:
                        model = CHARACTERISTICS[entry["Model Id"]]
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

    logging.info(f"Setting all techniques to Boolean values.")
    with tqdm(total=len(RESULTS)) as counter:
        for key, entry in RESULTS.items():
            for technique in TECHNIQUES:
                if technique not in entry:
                    entry[technique] = False
            counter.update(1)

    logging.info(f"Sorting data.")
    SIZE = len(RESULTS)
    DATA = {}
    TOOL_YEAR = {}
    with tqdm(total=len(RESULTS)) as counter:
        for _, entry in RESULTS.items():
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
                SIZE -= 1
                if entry["Clock Time"] < tool[entry["Year"]]["Clock Time"]:
                    tool[entry["Year"]] = entry
            else:
                tool[entry["Year"]] = entry
            counter.update(1)

    logging.info(f"Analyzing known data.")
    KNOWN = {}
    DISTANCE = ARGUMENTS.distance
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
                                subresults[tool]["time"] += entry["Clock Time"]
                                subresults[tool]["memory"] += entry["Memory"]
                            counter.update(1)
                    s = sorted(subsubresults.items(), key=lambda e: (
                        e[1]["time"], e[1]["memory"]))
                    # Select only the tools that are within a distance
                    # from the best:
                    known_i["sorted"] = [
                        {"tool": x[0],
                         "time": x[1]["time"],
                         "memory": x[1]["memory"]} for x in s]
                s = sorted(
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
                     "memory": x[1]["memory"]} for x in s]
                # Select all tools that reach both the maximum count and
                # the expected distance from the best:
                if known_m["sorted"]:
                    best = known_m["sorted"][0]
                    for x in known_m["sorted"]:
                        if x["count"] == best["count"]:
                            for instance, tools in instances.items():
                                tool = x["tool"]
                                ratio = x["time"] / best["time"]
                                if tool in tools \
                                        and TOOL_YEAR[tool] in tools[tool] \
                                        and (DISTANCE is None
                                             or ratio <= (1+DISTANCE)):
                                    entry = tools[tool][TOOL_YEAR[tool]]
                                    entry["Selected"] = True
    with open("known.json", "w") as output:
        json.dump(KNOWN, output)

    logging.info(f"Analyzing learned data.")
    LEARNED = []
    translate.ITEMS = {
        False: -1,
        None: 0,
        True: 1,
    }

    with tqdm(total=len(RESULTS)) as counter:
        for _, entry in RESULTS.items():
            if entry["Year"] == TOOL_YEAR[entry["Tool"]] \
                    and "Selected" in entry \
                    and entry["Selected"]:
                cp = {}
                for key, value in entry.items():
                    if key not in [
                            "Id", "Model Id", "Instance", "Year",
                            "Memory", "Clock Time",
                            "Parameterised", "Selected", "Surprise"] \
                            and key not in TECHNIQUES:
                        cp[key] = translate(value)
                LEARNED.append(cp)
            counter.update(1)
    logging.info(f"Select {len (LEARNED)} best entries.")
    # Convert this dict into dataframe:
    DF = pandas.DataFrame(LEARNED)
    # Remove duplicate entries if required:
    if not ARGUMENTS.duplicates:
        DF = DF.drop_duplicates(keep="first")
    logging.info(f"Using {DF.shape [0]} non duplicate entries for learning.")
    # Compute efficiency for each algorithm:
    ALGORITHMS_RESULTS = []
    for name, falgorithm in ALGORITHMS.items():
        subresults = []
        logging.info(f"Learning using algorithm: '{name}'.")
        for _ in tqdm(range(ARGUMENTS.iterations)):
            train, test = train_test_split(DF)
            training_X = train.drop("Tool", 1)
            training_Y = train["Tool"]
            test_X = test.drop("Tool", 1)
            test_Y = test["Tool"]
            # Apply algorithm:
            algorithm = falgorithm(True)
            algorithm.fit(training_X, training_Y)
            subresults.append(algorithm.score(test_X, test_Y))
        ALGORITHMS_RESULTS.append({
            "algorithm": name,
            "min": min(subresults),
            "max": max(subresults),
            "mean": statistics.mean(subresults),
            "median": statistics.median(subresults),
        })
        logging.info(f"Algorithm: {name}")
        logging.info(f"  Min     : {min                (subresults)}")
        logging.info(f"  Max     : {max                (subresults)}")
        logging.info(f"  Mean    : {statistics.mean    (subresults)}")
        logging.info(f"  Median  : {statistics.median  (subresults)}")
        logging.info(f"  Stdev   : {statistics.stdev   (subresults)}")
        logging.info(f"  Variance: {statistics.variance(subresults)}")
        algorithm = falgorithm(True)
        algorithm.fit(DF.drop("Tool", 1), DF["Tool"])
        with open(f"learned.{name}.p", "wb") as output:
            pickle.dump(algorithm, output)
    with open("learned.json", "w") as output:
        json.dump({
            "algorithms": ALGORITHMS_RESULTS,
            "translation": translate.ITEMS,
        }, output)
