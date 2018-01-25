#! /usr/bin/env python3

import argparse
import csv
import json
import logging
import os
import re
import statistics
import pandas
from tqdm                   import tqdm
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.ensemble       import BaggingClassifier
from sklearn.naive_bayes    import GaussianNB
from sklearn.svm            import SVC, LinearSVC
from sklearn                import tree
from sklearn.ensemble       import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

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

def knn(train_X, train_Y, test_X, test_Y):
    clf = KNeighborsClassifier(
        n_neighbors=10, weights='distance', algorithm='brute'
    )
    clf.fit(train_X, train_Y)
    return clf.score(test_X, test_Y)

def bagging_knn(train_X, train_Y, test_X, test_Y):
    # Bagging contains n_estimators of
    # KNeighborsClassifier(n_neighbors=10, weights='distance', n_jobs=4).
    # Every knn-classifier will take the half of the training set to learn.
    bagging = BaggingClassifier(
        KNeighborsClassifier(
            n_neighbors=10, weights='distance', algorithm='brute'
        ), max_samples=0.5, max_features=1, n_estimators=10
    )
    # Train the model:
    bagging.fit(train_X, train_Y)
    # Return the accuracy of guessing:
    return bagging.score(test_X, test_Y)

def gaussian_naive_bayes(train_X, train_Y, test_X, test_Y):
    # http://scikit-learn.org/stable/modules/naive_bayes.html
    gnb = GaussianNB()
    gnb.fit(train_X, train_Y)
    return gnb.score(test_X, test_Y)

def svm(train_X, train_Y, test_X, test_Y):
    # http://scikit-learn.org/stable/modules/svm.html
    # Support Vector Classifier with rbf kernel.
    clf = SVC()
    clf.fit(train_X, train_Y)
    return clf.score(test_X, test_Y)

def linear_svm(train_X, train_Y, test_X, test_Y):
    # http://scikit-learn.org/stable/modules/svm.html
    # Linear Support Vector classifier with rbf kernel.
    clf = LinearSVC()
    clf.fit(train_X, train_Y)
    return clf.score(test_X, test_Y)

def decision_tree(train_X, train_Y, test_X, test_Y):
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_X, train_Y)
    return clf.score(test_X, test_Y)

def random_forest(train_X, train_Y, test_X, test_Y):
    clf = RandomForestClassifier(n_estimators=10, max_features=None)
    clf = clf.fit(train_X, train_Y)
    return clf.score(test_X, test_Y)

def neural_net(train_X, train_Y, test_X, test_Y):
    # http://scikit-learn.org/stable/modules/neural_networks_supervised.html
    # Simple multi-layer neural net using Newton solver.
    nn = MLPClassifier(solver='lbfgs')
    nn.fit(train_X, train_Y)
    return nn.score(test_X, test_Y)

algorithms = {
    "KNN": knn,
    "Bagging+knn": bagging_knn,
    "Gaussian Naive Bayes": gaussian_naive_bayes,
    "Support Vector Machine": svm,
    "Linear Support Vector Machine": linear_svm,
    "Neural Net": neural_net,
    "Decision Tree": decision_tree,
    "Random Forest": random_forest,
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser (
        description = "..." # FIXME
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
        default = 100,
    )
    arguments = parser.parse_args ()
    logging.basicConfig (level = logging.INFO)

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
                    known_i ["sorted"] = [ { "tool": x [0], "time": x [1] ["time"], "memory": x [1] ["memory"] } for x in s]
                    rank = 0
                    for x in known_i ["sorted"]:
                        rank += 1
                        for tool, years in tools.items ():
                            for year, entry in years.items ():
                                if year == tool_year [tool]:
                                    entry ["Rank"] = rank
                s = sorted (subresults.items (), key = lambda e: (- e [1] ["count"], e [1] ["time"], e [1] ["memory"]))
                known_m ["sorted"] = [ { "tool": x [0], "count": x [1] ["count"], "time": x [1] ["time"], "memory": x [1] ["memory"] } for x in s]
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
            if entry ["Year"] == tool_year [entry ["Tool"]]:
                cp = {}
                for key, value in entry.items ():
                    if key != "Id" and key != "Model Id" and key != "Year" and key != "Instance":
                        cp [key] = translate (value)
                learned.append (cp)
            counter.update (1)
    with open ("learned.translation.json", "w") as output:
        json.dump (translation, output)

    # Convert this dict into dataframe:
    df = pandas.DataFrame (learned)
    # Select only the best tools:
    df = df.loc [df ["Rank"] == 1]
    # Remove the Tool columns from X.
    X = df.drop ("Tool", 1)
    Y = df ["Tool"]
    # Compute efficiency for each algorithm:
    for algorithm in algorithms:
        subresults = []
        print(f"  Algorithm: {algorithm}")
        for _ in tqdm (range (arguments.iterations)):
            n             = X.shape [0]
            training_size = int (n * 0.75)
            training_X    = X.sample (training_size)
            training_Y    = Y [training_X.index]
            # Remove the training points:
            tmp_X         = pandas.concat ([X, training_X]).drop_duplicates (keep = False)
            # Get the test points:
            test_X        = tmp_X.sample (min (int (n * 0.25), tmp_X.shape [0]))
            test_Y        = Y [test_X.index]
            subresults.append (algorithms.get (algorithm) (training_X, training_Y, test_X, test_Y))
        print(f"    Min     : {min                (subresults)}")
        print(f"    Max     : {max                (subresults)}")
        print(f"    Mean    : {statistics.mean    (subresults)}")
        print(f"    Median  : {statistics.median  (subresults)}")
        print(f"    Stdev   : {statistics.stdev   (subresults)}")
        print(f"    Variance: {statistics.variance(subresults)}")
