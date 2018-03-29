"""
Analysis of the results of the model checking contest.
"""

import logging
import pickle
import pandas
from tqdm import tqdm
from sklearn import tree
from mcc.model import Values, TECHNIQUES
from mcc.algorithms import ALGORITHMS

REMOVE = [
    "Id", "Model Id", "Model", "Instance", "Year",
    "Memory", "Time",
    "Parameterised", "Selected", "Surprise",
]


def score_of(data, alg_or_tool, values=None):
    """
    Computes a score using the rules from the MCC.
    """
    result = {}
    results = data.results()
    characteristics = data.characteristics()
    examinations = {x["Examination"] for x in results}
    models = [characteristics[id] for id in {x["Model Id"] for x in results}]
    tools = {x["Tool"] for x in results}
    with tqdm(total=len(examinations)*len(models)) as counter:
        for examination in examinations:
            result[examination] = 0
            for model in models:
                if alg_or_tool in tools:
                    tool = alg_or_tool
                elif isinstance(alg_or_tool, str):
                    tool = alg_or_tool
                else:
                    test = {}
                    if values is None:
                        values = Values(None)
                    test["Examination"] = values.to_learning(examination)
                    for key, value in model.items():
                        if key not in REMOVE \
                                and key not in TECHNIQUES:
                            test[key] = values.to_learning(value)
                    dataframe = pandas.DataFrame([test])
                    predicted = alg_or_tool.predict(dataframe)
                    tool = values.from_learning(predicted[0])
                subscore = 0
                instances = {
                    x["Instance"] for x in results
                    if x["Examination"] == examination
                    and x["Model"] == model
                }
                if not instances:
                    counter.update(1)
                    continue
                for instance in instances:
                    entries = [
                        x for x in results
                        if x["Examination"] == examination
                        and x["Model"] == model
                        and x["Instance"] == instance
                        and x["Tool"] == tool
                    ]
                    if entries:
                        subscore += 16
                subscore = subscore / len(instances)
                result[examination] = result[examination] + subscore
                counter.update(1)
    for examination, value in result.items():
        result[examination] = int(value)
    return result


def max_score(data):
    """
    Computes the maximum score using almost the rules from the MCC.
    """
    result = 0
    results = data.results()
    characteristics = data.characteristics()
    examinations = {x["Examination"] for x in results}
    models = [characteristics[id] for id in {x["Model Id"] for x in results}]
    for _ in examinations:
        for _ in models:
            result += 16
    return int(result)


def known(data):
    """
    Analyzes known data.
    """
    result = {}
    results = data.results()
    characteristics = data.characteristics()
    examinations = {x["Examination"] for x in results}
    tools = {x["Tool"] for x in results}
    models = [characteristics[id] for id in {x["Model Id"] for x in results}]
    instances = {x["Instance"] for x in results}
    with tqdm(total=len(examinations)*len(instances)) as counter:
        for examination in examinations:
            result[examination] = {}
            # Extract data for all known instances:
            for instance in instances:
                subresults = []
                for entry in [
                        x for x in results
                        if x["Examination"] == examination
                        and x["Instance"] == instance
                ]:
                    subresults.append({
                        "Time": entry["Time"],
                        "Memory": entry["Memory"],
                        "Tool": entry["Tool"],
                    })
                result[examination][instance] = sorted(
                    subresults,
                    key=lambda e: (e["Time"], e["Memory"], e["Tool"]),
                )
                counter.update(1)
    with tqdm(total=len(examinations)*len(models)) as counter:
        for examination in examinations:
            # Extract data for all known models:
            for model in models:
                all_instances = {
                    x["Instance"] for x in results
                    if x["Examination"] == examination
                    and x["Model"] == model
                }
                if not all_instances:
                    counter.update(1)
                    continue
                tools_results = []
                for tool in tools:
                    subresults = [
                        x for x in results
                        if x["Examination"] == examination
                        and x["Model"] == model
                        and x["Tool"] == tool
                    ]
                    count = 0
                    time = 0
                    memory = 0
                    for subresult in subresults:
                        count = count + 1
                        time = time + subresult["Time"]
                        memory = memory + subresult["Memory"]
                    tools_results.append({
                        "Count": count,
                        "Total": len(all_instances),
                        "Time": time,
                        "Memory": memory,
                        "Tool": tool,
                        "Ratio": count / len(all_instances),
                    })
                tools_results = sorted(
                    tools_results,
                    key=lambda e:
                    (-e["Ratio"], e["Time"], e["Memory"], e["Tool"]),
                )
                result[examination][model["Id"]] = tools_results
                counter.update(1)
    return result


def learned(data, options):
    """
    Analyzes learned data.
    """
    result = []
    results = data.results()
    characteristics = data.characteristics()
    examinations = {x["Examination"] for x in results}
    models = [characteristics[id] for id in {x["Model Id"] for x in results}]
    # For each examination and model, select only the good tools:
    with tqdm(total=len(examinations)*len(models)) as counter:
        for examination in examinations:
            for model in models:
                subresults = {}
                maximum = 0
                for entry in results:
                    if entry["Examination"] == examination \
                            and entry["Model"] == model:
                        tool = entry["Tool"]
                        if tool not in subresults:
                            subresults[tool] = set()
                        subresults[tool].add(entry["Instance"])
                        maximum = max(maximum, len(subresults[tool]))
                counter.update(1)
                for entry in results:
                    tool = entry["Tool"]
                    if "Selected" not in entry \
                            and entry["Examination"] == examination \
                            and entry["Model"] == model \
                            and len(subresults[tool]) == maximum:
                        entry["Selected"] = True
    # Extract selected entries and convert them to machine learning data:
    values = Values(None)
    selected = []
    with tqdm(total=len(results)) as counter:
        for entry in results:
            if "Selected" not in entry:
                counter.update(1)
                continue
            s_entry = {}
            for key, value in entry.items():
                if key not in REMOVE \
                        and key not in TECHNIQUES:
                    s_entry[key] = values.to_learning(value)
            selected.append(s_entry)
            counter.update(1)
    logging.info(f"Select {len (selected)} best entries of {len(results)}.")
    # Convert this dict into dataframe:
    dataframe = pandas.DataFrame(selected)
    # Drop extra fields in the dataframe:
    # Remove duplicate entries if required:
    if not options["Duplicates"]:
        dataframe = dataframe.drop_duplicates(keep="first")
        logging.info(f"Using {dataframe.shape [0]} non duplicate entries.")
    # Compute efficiency for each algorithm:
    for name, algorithm in ALGORITHMS.items():
        # Skip complex algorithms if duplicates data are allowed:
        if options["Duplicates"] and name in ["knn", "bagging-knn"]:
            continue
        logging.info(f"Learning using algorithm: '{name}'.")
        alg_results = {
            "Algorithm": name,
        }
        algorithm.fit(dataframe.drop("Tool", 1), dataframe["Tool"])
        score = score_of(data, algorithm, values)
        total = 0
        for key, value in score.items():
            alg_results[key] = value
            total = total + value
        logging.info(f"  Score: {total}")
        result.append(alg_results)
        with open(f"learned.{name}.p", "wb") as output:
            pickle.dump(algorithm, output)
        # Output decision tree and random forest to graphviz:
        if options["Output Trees"] \
                and name in ["decision-tree", "random-forest"]:
            tree.export_graphviz(
                algorithm,
                feature_names=dataframe.drop("Tool", 1).columns,
                filled=True,
                rounded=True,
                special_characters=True
            )
    return result
