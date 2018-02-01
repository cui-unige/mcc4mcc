#! /usr/bin/env python3

import argparse
import csv
import json
import logging
import os
import re
import statistics
import pickle
import pandas
from tqdm                               import tqdm
from sklearn.neighbors                  import KNeighborsClassifier
from sklearn.ensemble                   import BaggingClassifier, AdaBoostClassifier
from sklearn.naive_bayes                import GaussianNB
from sklearn.svm                        import SVC, LinearSVC
from sklearn                            import tree
from sklearn.ensemble                   import RandomForestClassifier
from sklearn.neural_network             import MLPClassifier
from sklearn.model_selection            import train_test_split

CHARACTERISTICS = [
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
RESULTS = [
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

def value_of (x):
    if x in [ "True", "Yes", "OK" ]:
        return True
    if x in [ "False", "None" ]:
        return False
    if x == "Unknown":
        return None
    try:
        return int (x)
    except ValueError:
        pass
    try:
        return float (x)
    except ValueError:
        pass
    return x

def knn_distance (x, y, bound = 2):
    # we suppose x and y have the same shape and are numpy array.
    diff = abs (x - y)
    diff [diff > bound] = bound
    return diff.sum ()

algorithms = {}

# algorithms ["knn"] = lambda _: KNeighborsClassifier (
#     n_neighbors = 10,
#     weights     = "distance",
#     metric      = knn_distance,
# )

# algorithms ["bagging-knn"] = lambda _: BaggingClassifier (
#     KNeighborsClassifier (
#         n_neighbors = 10,
#         weights     = "distance",
#         metric      = knn_distance
#     ),
#     max_samples  = 0.5,
#     max_features = 1,
#     n_estimators = 10,
# )

algorithms ["naive-bayes"] = lambda _: GaussianNB ()

algorithms ["svm"] = lambda _: SVC ()

algorithms ["ada boost"] = lambda _: AdaBoostClassifier()

algorithms ["linear-svm"] = lambda _: LinearSVC ()

algorithms ["decision-tree"] = lambda _: tree.DecisionTreeClassifier ()

algorithms ["random-forest"] = lambda _: RandomForestClassifier (
    n_estimators = 20,
    max_features = None,
)

algorithms ["neural-network"] = lambda _: MLPClassifier (
    solver = "lbfgs",
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser (
        description = "Data extractor for the model checker collection"
    )
    parser.add_argument (
        "--results",
        help    = "results of the model checking contest",
        type    = str,
        dest    = "results",
        default = os.getcwd () + "/results.csv",
    )
    parser.add_argument (
        "--characteristics",
        help    = "model characteristics from the Petri net repository",
        type    = str,
        dest    = "characteristics",
        default = os.getcwd () + "/characteristics.csv",
    )
    parser.add_argument (
        "--known",
        help    = "data known from models",
        type    = str,
        dest    = "known",
        default = os.getcwd () + "/known.json",
    )
    parser.add_argument (
        "--learned",
        help    = "data learned from models",
        type    = str,
        dest    = "learned",
        default = os.getcwd () + "/learned.json",
    )
    parser.add_argument(
        "--iterations",
        help    = "Number of iterations",
        type    = int,
        dest    = "iterations",
        default = 10,
    )
    parser.add_argument(
        "--distance",
        help    = "Allowed distance from the best tool (in percent)",
        type    = float,
        dest    = "distance",
    )
    arguments = parser.parse_args ()
    logging.basicConfig (
        level  = logging.INFO,
        format = "%(levelname)s: %(message)s",
    )

    techniques = {}

    characteristics = {}
    logging.info (f"Reading model characteristics from '{arguments.characteristics}'.")
    with tqdm (total = sum (1 for line in open (arguments.characteristics)) - 1) as counter:
        with open (arguments.characteristics) as data:
            data.readline () # skip the title line
            reader = csv.reader (data)
            for row in reader:
                entry = {}
                for i, characteristic in enumerate (CHARACTERISTICS):
                    entry [characteristic] = value_of (row [i])
                entry ["Place/Transition"] = True if re.search ("PT"     , entry ["Type"]) else False
                entry ["Colored"         ] = True if re.search ("COLORED", entry ["Type"]) else False
                del entry ["Type"      ]
                del entry ["Fixed size"]
                del entry ["Origin"    ]
                del entry ["Submitter" ]
                del entry ["Year"      ]
                characteristics [entry ["Id"]] = entry
                counter.update (1)

    results = {}
    logging.info (f"Reading mcc results from '{arguments.results}'.")
    with tqdm (total = sum (1 for line in open (arguments.results)) - 1) as counter:
        with open (arguments.results) as data:
            data.readline () # skip the title line
            reader = csv.reader (data)
            for row in reader:
                entry = {}
                for i, result in enumerate (RESULTS):
                    entry [result] = value_of (row [i])
                if  entry ["Time OK"  ] \
                and entry ["Memory OK"] \
                and entry ["Status"   ] == "normal" \
                and entry ["Results"  ] != "DNC" \
                and entry ["Results"  ] != "DNF" \
                and entry ["Results"  ] != "CC":
                    results [entry ["Id"]] = entry
                    for technique in re.findall (r"([A-Z_]+)", entry ["Techniques"]):
                        techniques [technique] = True
                        entry [technique] = True
                    entry ["Surprise"] = True if re.search (r"^S_", entry ["Instance"]) else False
                    if entry ["Surprise"]:
                        entry ["Instance"] = re.search (r"^S_(.*)$", entry ["Instance"]).group (1)
                    split = re.search (r"([^-]+)\-([^-]+)\-([^-]+)$", entry ["Instance"])
                    if split is None:
                        entry ["Model Id"] = entry ["Instance"]
                    else:
                        entry ["Model Id"] = split.group (1)
                    if entry ["Model Id"] in characteristics:
                        model = characteristics [entry ["Model Id"]]
                        for key in model.keys ():
                            if key != "Id":
                                entry [key] = model [key]
                    del entry ["Time OK"   ]
                    del entry ["Memory OK" ]
                    del entry ["CPU Time"  ]
                    del entry ["Cores"     ]
                    del entry ["IO Time"   ]
                    del entry ["Results"   ]
                    del entry ["Status"    ]
                    del entry ["Techniques"]
                counter.update (1)

    logging.info (f"Setting all techniques to Boolean values.")
    with tqdm (total = len (results)) as counter:
        for key, entry in results.items ():
            for technique in techniques:
                if technique not in entry:
                    entry [technique] = False
            counter.update (1)

    logging.info (f"Sorting data.")
    size      = len (results)
    data      = {}
    tool_year = {}
    with tqdm (total = len (results)) as counter:
        for _, entry in results.items ():
            if entry ["Examination"] not in data:
                data [entry ["Examination"]] = {}
            examination = data [entry ["Examination"]]
            if entry ["Model Id"] not in examination:
                examination [entry ["Model Id"]] = {}
            model = examination [entry ["Model Id"]]
            if entry ["Instance"] not in model:
                model [entry ["Instance"]] = {}
            instance = model [entry ["Instance"]]
            if entry ["Tool"] not in instance:
                instance [entry ["Tool"]] = {}
            tool = instance [entry ["Tool"]]
            if entry ["Tool"] not in tool_year:
                tool_year [entry ["Tool"]] = 0
            if entry ["Year"] > tool_year [entry ["Tool"]]:
                tool_year [entry ["Tool"]] = entry ["Year"]
            if entry ["Year"] in tool:
                size -= 1
                if entry ["Clock Time"] < tool [entry ["Year"]] ["Clock Time"]:
                    tool [entry ["Year"]] = entry
            else:
                tool [entry ["Year"]] = entry
            counter.update (1)

    logging.info (f"Analyzing known data.")
    known = {}
    distance = arguments.distance
    with tqdm (total = size) as counter:
        for examination, models in data.items ():
            known [examination] = {}
            known_e = known [examination]
            for model, instances in models.items ():
                known_e [model] = {}
                known_m = known_e [model]
                subresults = {}
                for instance, tools in instances.items ():
                    known_m [instance] = {}
                    known_i = known_m [instance]
                    subsubresults = {}
                    for tool, years in tools.items ():
                        if tool not in subresults:
                            subresults [tool] = {
                                "count" : 0,
                                "time"  : 0,
                                "memory": 0,
                            }
                        for year, entry in years.items ():
                            if year == tool_year [tool]:
                                subsubresults [tool] = {
                                    "time"  : entry ["Clock Time"],
                                    "memory": entry ["Memory"],
                                }
                                subresults [tool] ["count"]  += 1
                                subresults [tool] ["time"]   += entry ["Clock Time"]
                                subresults [tool] ["memory"] += entry ["Memory"]
                            counter.update (1)
                    s = sorted (subsubresults.items (), key = lambda e: (e [1] ["time"], e [1] ["memory"]))
                    # Select only the tools that are within a distance from the best:
                    known_i ["sorted"] = [ { "tool": x [0], "time": x [1] ["time"], "memory": x [1] ["memory"] } for x in s]
                s = sorted (subresults.items (), key = lambda e: (- e [1] ["count"], e [1] ["time"], e [1] ["memory"]))
                known_m ["sorted"] = [ { "tool": x [0], "count": x [1] ["count"], "time": x [1] ["time"], "memory": x [1] ["memory"] } for x in s]
                # Select all tools that reach both the maximum count and the expected distance from the best:
                if known_m ["sorted"]:
                    best = known_m ["sorted"] [0]
                    for x in known_m ["sorted"]:
                        if x ["count"] == best ["count"]:
                            for instance, tools in instances.items ():
                                tool  = x ["tool"]
                                if  tool in tools \
                                and tool_year [tool] in tools [tool] \
                                and (distance is None or x ["time"] / best ["time"] <= (1+distance)):
                                    entry = tools [tool] [tool_year [tool]]
                                    entry ["Selected"] = True
    with open ("known.json", "w") as output:
        json.dump (known, output)

    logging.info (f"Analyzing learned data.")
    learned     = []
    next_id     = 10
    translation = {
        False: -1,
        None : 0,
        True : 1,
    }
    def translate (x):
        global next_id
        if x is None:
            return 0
        if isinstance (x, (bool, str)):
            if x not in translation:
                translation [x] = next_id + 1
                next_id        += 1
            return translation [x]
        else:
            return x
    with tqdm (total = len (results)) as counter:
        for _, entry in results.items ():
            if  entry ["Year"] == tool_year [entry ["Tool"]] \
            and "Selected" in entry \
            and entry ["Selected"]:
                cp = {}
                for key, value in entry.items ():
                    if  key != "Id" \
                    and key != "Model Id" \
                    and key != "Year" \
                    and key != "Instance" \
                    and key != "Memory" \
                    and key != "Clock Time" \
                    and key != "Parameterised" \
                    and key != "Selected" \
                    and key != "Surprise" \
                    and key not in techniques:
                        cp [key] = translate (value)
                learned.append (cp)
            counter.update (1)
    logging.info (f"Using {len (learned)} entries for learning.")
    # Convert this dict into dataframe:
    df = pandas.DataFrame (learned)
    # Compute efficiency for each algorithm:
    algorithms_results = []
    for name, falgorithm in algorithms.items ():
        subresults = []
        logging.info (f"Learning using algorithm: '{name}'.")
        for _ in tqdm (range (arguments.iterations)):
            train, test = train_test_split(df)
            training_X  = train.drop("Tool", 1)
            training_Y  = train["Tool"]
            test_X      = test.drop("Tool", 1)
            test_Y      = test["Tool"]
            # Apply algorithm:
            algorithm   = falgorithm (True)
            algorithm.fit (training_X, training_Y)
            subresults.append (algorithm.score (test_X, test_Y))
        algorithms_results.append ({
            "algorithm": name,
            "min"      : min (subresults),
            "max"      : max (subresults),
            "mean"     : statistics.mean (subresults),
            "median"   : statistics.median (subresults),
        })
        logging.info (f"Algorithm: {name}")
        logging.info (f"  Min     : {min                (subresults)}")
        logging.info (f"  Max     : {max                (subresults)}")
        logging.info (f"  Mean    : {statistics.mean    (subresults)}")
        logging.info (f"  Median  : {statistics.median  (subresults)}")
        logging.info (f"  Stdev   : {statistics.stdev   (subresults)}")
        logging.info (f"  Variance: {statistics.variance(subresults)}")
        algorithm = falgorithm (True)
        algorithm.fit (df.drop("Tool", 1), df["Tool"])
        with open (f"learned.{name}.p", "wb") as output:
            pickle.dump (algorithm, output)
    with open ("learned.json", "w") as output:
        json.dump ({
            "algorithms" : algorithms_results,
            "translation": translation,
        }, output)
