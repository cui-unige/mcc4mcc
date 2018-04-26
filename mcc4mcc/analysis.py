"""
Analysis of the results of the model checking contest.
"""

import logging
import pickle
import statistics
from random import shuffle
import pandas
from frozendict import frozendict
from tqdm import tqdm
from sklearn import tree
from mcc4mcc.model import Values, TECHNIQUES
from mcc4mcc.algorithms import ALGORITHMS

REMOVE = [
    "Id", "Model", "Instance", "Year",
    "Memory", "Time",
    "Parameterised", "Selected", "Surprise",
]


def characteristics_of(data):
    """
    Computes the average models per characteristics set.
    """
    result = {}
    all_characteristics = set({})
    for _, model in data.characteristics().items():
        for key in model.keys():
            if key not in data.configuration["forget"]:
                all_characteristics.add(key)
    logging.info(f"Characteristics taken into account are:")
    for char in sorted(all_characteristics):
        if char != "Id":
            logging.info(f"  * {char}")
    result = {}
    for identifier, model in data.characteristics().items():
        stripped = dict(model)
        for characteristic in data.configuration["forget"]:
            del stripped[characteristic]
        if "Id" in stripped:
            del stripped["Id"]
        stripped = frozendict(stripped)
        if stripped not in result:
            result[stripped] = []
        result[stripped].append(identifier)
    stats = statistics.mean([len(v) for k, v in result.items()])
    logging.info(f"Mean models per characteristics set: {stats}.")
    return result


# def characteristics_of(characteristics):
#     """
#     Computes the interesting characteristics to remove.
#     """
#     result = {}
#     all_characteristics = set({})
#     for _, model in characteristics.items():
#         for key in model.keys():
#             all_characteristics.add(key)
#     logging.info(f"Characteristics are:")
#     for char in sorted(all_characteristics):
#         logging.info(f"  * {char}")
#     logging.info(f"  (Id is never taken into account)")
#     logging.info(f"Computing models per characteristics set.")
#     all_subsets = [x for x in powerset(all_characteristics)]
#     with tqdm(total=len(all_subsets)) as counter:
#         for subset in all_subsets:
#             subresult = {}
#             for id, model in characteristics.items():
#                 stripped = dict(model)
#                 for characteristic in subset:
#                     del stripped[characteristic]
#                 if "Id" in stripped:
#                     del stripped["Id"]
#                 stripped = frozendict(stripped)
#                 if stripped not in subresult:
#                     subresult[stripped] = []
#                 subresult[stripped].append(id)
#             result[subset] = subresult
#             stats = statistics.mean([len(v) for k,v in subresult.items()])
#             if stats > 2:
#                 logging.info(f"  * {subset} ({stats})")
#             counter.update(1)
#     return result


def choice_of(data, alg_or_tool, values=None):
    """
    Computes for each examination the repartition of choices.
    """
    result = {}
    results = data.results()
    data.characteristics()
    examinations = {x["Examination"] for x in results}
    models = {x["Model"] for x in results}
    tools = {x["Tool"] for x in results}
    with tqdm(total=len(examinations)*len(models)) as counter:
        for examination in examinations:
            for_examination = [
                x for x in results
                if x["Examination"] == examination
            ]
            result[examination] = {}
            for tool in tools:
                result[examination][tool] = 0
            for model in models:
                for_model = [
                    x for x in for_examination
                    if x["Model"] is model
                ]
                if not for_model:
                    counter.update(1)
                    continue
                if alg_or_tool in tools:
                    tool = alg_or_tool
                elif isinstance(alg_or_tool, str):
                    tool = alg_or_tool
                else:
                    test = {}
                    if values is None:
                        values = Values(None)
                    test["Examination"] = values.to_learning(examination)
                    test["Relative-Time"] = 1  # FIXME
                    test["Relative-Memory"] = 1  # FIXME
                    for key, value in model.items():
                        if key in data.configuration["forget"]:
                            test[key] = values.to_learning(None)
                        elif key not in REMOVE \
                                and key not in TECHNIQUES:
                            test[key] = values.to_learning(value)
                    dataframe = pandas.DataFrame([test])
                    predicted = alg_or_tool.predict(dataframe)
                    tool = values.from_learning(predicted[0])
                result[examination][tool] += 1
                counter.update(1)
    return result


def score_of(data, alg_or_tool, values=None):
    """
    Computes a score using the rules from the MCC.
    """
    result = {}
    results = data.results()
    data.characteristics()
    examinations = {x["Examination"] for x in results}
    models = {x["Model"] for x in results}
    tools = {x["Tool"] for x in results}
    with tqdm(total=len(examinations)*len(models)) as counter:
        for examination in examinations:
            for_examination = [
                x for x in results
                if x["Examination"] == examination
            ]
            result[examination] = 0
            for model in models:
                for_model = [
                    x for x in for_examination
                    if x["Model"] is model
                ]
                if not for_model:
                    counter.update(1)
                    continue
                if alg_or_tool in tools:
                    tool = alg_or_tool
                elif isinstance(alg_or_tool, str):
                    tool = alg_or_tool
                else:
                    test = {}
                    if values is None:
                        values = Values(None)
                    test["Examination"] = values.to_learning(examination)
                    test["Relative-Time"] = values.to_learning(1)  # FIXME
                    test["Relative-Memory"] = values.to_learning(1)  # FIXME
                    for key, value in model.items():
                        if key in data.configuration["forget"]:
                            test[key] = values.to_learning(None)
                        elif key not in REMOVE \
                                and key not in TECHNIQUES:
                            test[key] = values.to_learning(value)
                    dataframe = pandas.DataFrame([test])
                    predicted = alg_or_tool.predict(dataframe)
                    tool = values.from_learning(predicted[0])
                subscore = 0
                instances = {
                    x["Instance"] for x in for_model
                }
                for instance in instances:
                    entries = [
                        x for x in for_model
                        if x["Instance"] == instance
                        and x["Tool"] == tool
                    ]
                    if entries:
                        subscore += 16
                    best_time = [
                        x for x in entries
                        if x["Relative-Time"] == 1
                    ]
                    if best_time:
                        subscore += 2
                    best_memory = [
                        x for x in entries
                        if x["Relative-Memory"] == 1
                    ]
                    if best_memory:
                        subscore += 2
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
    data.characteristics()
    examinations = {x["Examination"] for x in results}
    models = {x["Model"] for x in results}
    for _ in examinations:
        for _ in models:
            result += 20
    return int(result)


def known(data):
    """
    Analyzes known data.
    """
    result = {}
    results = data.results()
    data.characteristics()
    examinations = {x["Examination"] for x in results}
    tools = {x["Tool"] for x in results}
    models = {x["Model"] for x in results}
    instances = {x["Instance"] for x in results}
    logging.info(
        f"Analyzing known data."
    )
    with tqdm(total=len(examinations)*len(instances)) as counter:
        for examination in examinations:
            for_examination = [
                x for x in results
                if x["Examination"] == examination
            ]
            result[examination] = {}
            # Extract data for all known instances:
            for instance in instances:
                subresults = []
                for entry in [
                        x for x in for_examination
                        if x["Instance"] == instance
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
            for_examination = [
                x for x in results
                if x["Examination"] == examination
            ]
            # Extract data for all known models:
            for model in models:
                for_model = [
                    x for x in for_examination
                    if x["Model"] is model
                ]
                all_instances = {
                    x["Instance"] for x in for_model
                }
                if not all_instances:
                    counter.update(1)
                    continue
                tools_results = []
                for tool in tools:
                    subresults = [
                        x for x in for_model
                        if x["Tool"] == tool
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
    directory = options["Directory"]
    prefix = options["Prefix"]
    result = []
    results = data.results()
    data.characteristics()
    examinations = {x["Examination"] for x in results}
    models = {x["Model"] for x in results}
    # For each examination and model, select only the good tools:
    logging.info(
        f"Analyzing learned data."
    )
    # Keep only some models into a training and a test set:
    all_models = list(models)
    shuffle(all_models)
    training = all_models[:int(len(models)*options["Training"])]
    if len(training) != len(models):
        logging.info(
            f"  Keeping only {len(training)} models of {len(models)}."
        )
    # Select entries:
    selected = set()
    with tqdm(total=len(examinations)*len(training)) as counter:
        for examination in examinations:
            for model in training:
                subresults = {}
                maximum = 0
                for entry in results:
                    if entry["Examination"] == examination \
                            and entry["Model"] is model:
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
                            and entry["Model"] is model \
                            and len(subresults[tool]) == maximum:
                        selected.add(entry)
    # Extract selected entries and convert them to machine learning data:
    values = Values(None)
    selection = []
    with tqdm(total=len(selected)) as counter:
        for entry in selected:
            s_entry = {}
            for key, value in entry.items():
                if key in data.configuration["forget"]:
                    s_entry[key] = values.to_learning(None)
                elif key not in REMOVE \
                        and key not in TECHNIQUES:
                    s_entry[key] = values.to_learning(value)
            for key, value in entry["Model"].items():
                if key in data.configuration["forget"]:
                    s_entry[key] = values.to_learning(None)
                elif key not in REMOVE:
                    s_entry[key] = values.to_learning(value)
            selection.append(s_entry)
            counter.update(1)
    logging.info(f"Select {len (selection)} best entries of {len(results)}.")
    # Convert this dict into dataframe:
    dataframe = pandas.DataFrame(selection)
    # Remove duplicate entries if required:
    if not options["Duplicates"]:
        dataframe = dataframe.drop_duplicates(keep="first")
        logging.info(f"Using {dataframe.shape [0]} non duplicate entries.")
    # Compute efficiency for each algorithm:
    for alg_entry in sorted(ALGORITHMS.items(), key=lambda e: e[0]):
        name = alg_entry[0]
        algorithm = alg_entry[1](None)
        # Skip complex algorithms if duplicates data are allowed:
        if options["Duplicates"] and name in ["knn", "bagging-knn"]:
            continue
        logging.info(f"Learning using algorithm: {name}.")
        alg_results = {
            "Algorithm": name,
            "Is-Tool": False,
            "Is-Algorithm": True,
        }
        algorithm.fit(dataframe.drop("Tool", 1), dataframe["Tool"])
        # Compute score:
        score = score_of(data, algorithm, values)
        total = 0
        for key, value in score.items():
            alg_results[key] = value
            total = total + value
        logging.info(f"  Score: {total}")
        result.append(alg_results)
        # Compute choice:
        choice = choice_of(data, algorithm, values)
        for examination in sorted(examinations):
            logging.info(f"  In {examination}:")
            srt = sorted(
                choice[examination].items(),
                key=lambda e: e[1],
                reverse=True
            )
            for entry in srt:
                tool = entry[0]
                value = entry[1]
                if value > 0:
                    logging.info(f"  * {tool} is chosen {value} times")
        # Store algorithm:
        with open(f"{directory}/{prefix}-learned.{name}.p", "wb") as output:
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
    return result, values
