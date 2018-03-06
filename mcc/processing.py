"""
All kind of preprocessing functions. It helps to transform and filter the
machine learning datas.
"""

import csv
import re
import json
from tqdm import tqdm


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


def read_characteristics(arguments, g_v):
    """
    Reads the model characteristics.
    """
    with tqdm(total=sum(
        1 for line in open(arguments.characteristics)) - 1) \
            as counter:
        with open(arguments.characteristics) as data:
            data.readline()  # skip the title line
            reader = csv.reader(data)
            for row in reader:
                entry = {}
                for i, characteristic in enumerate(g_v.CHARACTERISTIC_KEYS):
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
                g_v.CHARACTERISTICS[entry["Id"]] = entry
                counter.update(1)


def read_results(arguments, g_v):
    """
    Reads the results of the model checking contest.
    """
    with tqdm(total=sum(1 for line in open(arguments.results)) - 1) \
            as counter:
        with open(arguments.results) as data:
            data.readline()  # skip the title line
            reader = csv.reader(data)
            for row in reader:
                entry = {}
                for i, result in enumerate(g_v.RESULT_KEYS):
                    entry[result] = value_of(row[i])
                if entry["Time OK"] \
                        and entry["Memory OK"] \
                        and entry["Status"] == "normal" \
                        and entry["Results"] not in ["DNC", "DNF", "CC"]:
                    g_v.RESULTS[entry["Id"]] = entry
                    for technique in re.findall(
                            r"([A-Z_]+)",
                            entry["Techniques"]
                    ):
                        g_v.TECHNIQUES[technique] = True
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
                    if entry["Model Id"] in g_v.CHARACTERISTICS:
                        model = g_v.CHARACTERISTICS[entry["Model Id"]]
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


def set_techniques(g_v):
    """
    Sets techniques to Boolean values in results.
    """
    with tqdm(total=len(g_v.RESULTS)) as counter:
        for _, entry in g_v.RESULTS.items():
            for technique in g_v.TECHNIQUES:
                if technique not in entry:
                    entry[technique] = False
            counter.update(1)


def rename_tools(g_v):
    """
    Rename tools that are duplicated.
    """
    with tqdm(total=len(g_v.RESULTS)) as counter:
        for _, entry in g_v.RESULTS.items():
            name = entry["Tool"]
            if name in g_v.TOOLS_RENAME:
                entry["Tool"] = g_v.TOOLS_RENAME[name]
            counter.update(1)


def sort_data(g_v):
    """
    Sorts data into tree of examination/model/instance/tool/year/entry.
    """
    size = g_v.SIZE
    with tqdm(total=len(g_v.RESULTS)) as counter:
        for _, entry in g_v.RESULTS.items():
            if entry["Tool"] not in g_v.TOOLS:
                g_v.TOOLS[entry["Tool"]] = True
            if entry["Examination"] not in g_v.DATA:
                g_v.DATA[entry["Examination"]] = {}
            examination = g_v.DATA[entry["Examination"]]
            if entry["Model Id"] not in examination:
                examination[entry["Model Id"]] = {}
            model = examination[entry["Model Id"]]
            if entry["Instance"] not in model:
                model[entry["Instance"]] = {}
            instance = model[entry["Instance"]]
            if entry["Tool"] not in instance:
                instance[entry["Tool"]] = {}
            tool = instance[entry["Tool"]]
            if entry["Tool"] not in g_v.TOOL_YEAR:
                g_v.TOOL_YEAR[entry["Tool"]] = 0
            if entry["Year"] > g_v.TOOL_YEAR[entry["Tool"]]:
                g_v.TOOL_YEAR[entry["Tool"]] = entry["Year"]
            if entry["Year"] in tool:
                size -= 1
                if entry["Clock Time"] < tool[entry["Year"]]["Clock Time"]:
                    tool[entry["Year"]] = entry
            else:
                tool[entry["Year"]] = entry
            counter.update(1)
    return size


def analyze_known(g_v):
    """
    Analyzes known data.
    """
    with tqdm(total=g_v.SIZE) as counter:
        for examination, models in g_v.DATA.items():
            g_v.KNOWN[examination] = {}
            known_e = g_v.KNOWN[examination]
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
                            if year == g_v.TOOL_YEAR[tool]:
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
                                if (
                                        tool in tools and g_v.TOOL_YEAR[tool]
                                        in tools[tool] and (
                                            g_v.DISTANCE is None or ratio <= (
                                                1 + g_v.DISTANC)
                                        )
                                ):
                                    entry = tools[tool][g_v.TOOL_YEAR[tool]]
                                    entry["Selected"] = True
    with open("known.json", "w") as output:
        json.dump(g_v.KNOWN, output)
