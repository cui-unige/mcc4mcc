"""
Analysis of the results of the model checking contest.
"""

import json
import logging
import statistics
import pickle
import pandas
from sklearn.model_selection import train_test_split
from sklearn import tree
from tqdm import tqdm
from mcc.model import Values


def score_of(data, alg_or_tool, to_drop=None):
    """
    Computes a score using the rules from the MCC.
    """
    result = {}
    results = data.results()
    characteristics = data.characteristics()
    examinations = {x["Examination"] for x in results}
    models = [characteristics[id] for id in {x["Model Id"] for x in results}]
    tools = {x["Tool"] for x in results}
    for examination in examinations():
        result[examination] = 0
        for model in models():
            if alg_or_tool in tools:
                tool = alg_or_tool
            else:
                test = {}
                values = Values(None)
                test["Examination"] = values.to_learning(examination)
                for key, value in model.items():
                    test[key] = values.to_learning(value)
                del test["Id"]
                del test["Parameterised"]
                dataframe = pandas.DataFrame([test])
                if to_drop is not None:
                    dataframe = dataframe.drop(to_drop, 1)
                tool = values.from_learning(alg_or_tool.predict(dataframe)[0])
            subscore = 0
            instances = {
                x["Instance"] for x in results
                if x["Examination"] == examination
                and x["Model"] == model
            }
            for instance in instances():
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
    for examination, value in result.items():
        result[examination] = int(value)
    return result


def scores(data):
    """
    Computes the scores of all tools.
    """
    result = {}
    results = data.results()
    tools = {x["Tool"] for x in results}
    with tqdm(total=len(tools)) as counter:
        for tool in tools:
            result[tool] = score_of(data, tool)
            counter.update(1)
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
        for _ in models():
            result += 16 + 2 + 2
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
                if all_instances:
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


# def analyze_learned(arguments, data):
#     """
#     Analyzes learned data.
#     """
#     with tqdm(total=len(data.results)) as counter:
#         for _, entry in data.results.items():
#             if entry["Year"] == data.tool_year[entry["Tool"]] \
#                     and "Selected" in entry \
#                     and entry["Selected"]:
#                 characteristics = {}
#                 for key, value in entry.items():
#                     if key not in data.remove \
#                             and key not in data.techniques:
#                         characteristics[key] = translate(value)
#                 data.learned.append(characteristics)
#             counter.update(1)
#     logging.info(f"Select {len (data.learned)} best entries.")
#     # Convert this dict into dataframe:
#     dataframe = pandas.DataFrame(data.learned)
#     # Remove duplicate entries if required:
#     if not arguments.duplicates:
#         dataframe = dataframe.drop_duplicates(keep="first")
#     logging.info(f"Using {dataframe.shape [0]} non duplicate entries.")
#     # Compute efficiency for each algorithm:
#     for name, algorithm in data.algorithms.items():
#         subresults = []
#         logging.info(f"Learning using algorithm: '{name}'.")
#         alg_results = {
#             "algorithm": name,
#         }
#         if arguments.iterations > 0:
#             for _ in tqdm(range(arguments.iterations)):
#                 train, test = train_test_split(dataframe)
#                 training_x = train.drop("Tool", 1)
#                 training_y = train["Tool"]
#                 test_x = test.drop("Tool", 1)
#                 test_y = test["Tool"]
#                 # Apply algorithm:
#                 algorithm.fit(training_x, training_y)
#                 subresults.append(algorithm.score(test_x, test_y))
#             alg_results["min"] = min(subresults)
#             alg_results["max"] = max(subresults)
#             alg_results["mean"] = statistics.mean(subresults)
#             alg_results["median"] = statistics.median(subresults)
#             logging.info(f"Algorithm: {name}")
#             logging.info(f"  Min     : {min                (subresults)}")
#             logging.info(f"  Max     : {max                (subresults)}")
#             logging.info(f"  Mean    : {statistics.mean    (subresults)}")
#             logging.info(f"  Median  : {statistics.median  (subresults)}")
#         algorithm.fit(dataframe.drop("Tool", 1), dataframe["Tool"])
#         if arguments.mcc_score:
#             data.scores[name] = mcc_score(algorithm, data)
#             for key, value in data.scores[name].items():
#                 alg_results[key] = value
#             total = data.scores[name]["Total"]
#             logging.info(f"  Score   : {total}")
#         data.algorithms_results.append(alg_results)
#         with open(f"learned.{name}.p", "wb") as output:
#             pickle.dump(algorithm, output)
#     with open("learned.json", "w") as output:
#         json.dump({
#             "algorithms": data.algorithms_results,
#             "translation": translate.ITEMS,
#         }, output)
#     if arguments.mcc_score:
#         logging.info(f"Maximum score is {max_score(data)}.")
#         srt = []
#         for name, score in data.scores.items():
#             for examination, value in score.items():
#                 srt.append({
#                     "name": name,
#                     "examination": examination,
#                     "score": value,
#                 })
#         srt = sorted(srt, key=lambda e: (
#             e["examination"], e["score"], e["name"]
#         ), reverse=True)
#         for element in srt:
#             examination = element["examination"]
#             score = element["score"]
#             name = element["name"]
#             logging.info(f"In {examination} : {score} for {name}.")
#     if arguments.output_dt:
#         if "decision-tree" in data.algorithms:
#             tree.export_graphviz(
#                 data.algorithms["decision-tree"](True).fit(
#                     dataframe.drop("Tool", 1), dataframe["Tool"]
#                 ),
#                 feature_names=dataframe.drop("Tool", 1).columns,
#                 filled=True, rounded=True,
#                 special_characters=True
#             )
#
#
# def analyze_useless(arguments, data):
#     """
#     Analyzes useless characteristics.
#     """
#     # Build the dataframe:
#     learned = []
#     with tqdm(total=len(data.results)) as counter:
#         for _, entry in data.results.items():
#             if entry["Year"] == data.tool_year[entry["Tool"]] \
#                     and "Selected" in entry \
#                     and entry["Selected"]:
#                 characteristics = {}
#                 for key, value in entry.items():
#                     if key not in data.remove \
#                             and key not in data.techniques:
#                         characteristics[key] = translate(value)
#                 learned.append(characteristics)
#             counter.update(1)
#     # Convert this dict into dataframe:
#     dataframe = pandas.DataFrame(learned)
#     # Remove duplicate entries if required:
#     if not arguments.duplicates:
#         dataframe = dataframe.drop_duplicates(keep="first")
#     logging.info(f"Using {dataframe.shape [0]} non duplicate entries.")
#     results = {}
#     # For each algorithm, try to drop each characteristic,
#     # and compare the score with the same with all characteristics:
#     for name, falgorithm in data.algorithms.items():
#         useless = {}
#         for characteristic in data.to_drop:
#             useless[characteristic] = True
#         logging.info(f"Analyzing characteristics in algorithm {name}.")
#         results[name] = {}
#         with tqdm(total=len(data.to_drop)) as counter:
#             for to_drop in data.to_drop:
#                 algorithm = falgorithm(True)
#                 algorithm.fit(
#                     dataframe.drop("Tool", 1).drop(to_drop, 1),
#                     dataframe["Tool"]
#                 )
#                 score = mcc_score(algorithm, data, to_drop)
#                 # If the score has changed,
#                 # the characteristic is not useless:
#                 if score != data.scores[name]:
#                     useless[to_drop] = False
#                 counter.update(1)
#         # The set of potential useless characteristics is obtained:
#         useless = set([x for x, y in useless.items() if y])
#         # If empty, there is no need for further investigation:
#         if useless is None:
#             return
#         logging.info(f"  Some characteristics in {useless} are useless.")
#         all_related = []
#         # Try to find which characteristics are truly useless,
#         # and which ones are linked to others.
#         # To do so, build tuples of n characteristics (n growing from 2),
#         # and try to remove them from the model.
#         for length in range(2, len(useless)):
#             sets = [list(x) for x in powerset(useless) if len(x) == length]
#             related = []
#             with tqdm(total=len(sets)) as counter:
#                 for characteristics in sets:
#                     # If a part of the tuple is already linked,
#                     # there is no need for investigation,
#                     # as the result will be linked:
#                     is_interesting = True
#                     for rel in all_related:
#                         if rel < set(characteristics):
#                             is_interesting = False
#                             break
#                     if is_interesting:
#                         algorithm = falgorithm(True)
#                         algorithm.fit(
#                             dataframe.drop("Tool", 1)
#                             .drop(characteristics, 1),
#                             dataframe["Tool"]
#                         )
#                         score = mcc_score(algorithm, data, characteristics)
#                         # If the score has changed, the characteristics
#                         # are linked:
#                         if score != data.scores[name]:
#                             related.append(set(characteristics))
#                     counter.update(1)
#             if not related:
#                 break
#             for rel in related:
#                 logging.info(f"  Characteristics {rel} are related.")
#                 all_related.append(rel)
#         # Characteristics that are linked to none are truly useless:
#         for rel in all_related:
#             useless -= rel
#         logging.info(f"  Characteristics {useless} are useless.")
