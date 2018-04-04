"""
Extract data about models.
"""

import logging
import csv
import re
import itertools
from tqdm import tqdm

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

TECHNIQUES = []


def value_of(what):
    """
    Converts a string, such as True, Yes, ... to its real value.
    """
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


class Values:
    """
    Conversion between values in models and learning algorithms.
    """
    def __init__(self, items):
        if items is None:
            self.items = {
                False: -1,
                None: 0,
                True: 1,
            }
            self.next_id = 10
        else:
            self.items = items
            self.next_id = sorted(items.values(), reverse=True)[0]

    def to_learning(self, what):
        """
        Translates values into numbers for machine learning algorithms.
        """
        if what is None:
            return 0
        if isinstance(what, (bool, str)):
            if what not in self.items:
                self.items[what] = self.next_id + 1
                self.next_id += 1
            return self.items[what]
        else:
            return what

    def from_learning(self, what):
        """
        Translate values from machine learning to their initial value.
        """
        for key, value in self.items.items():
            if value == what:
                return key
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


class Data:
    """
    Data from the model checking contest.
    """
    def __init__(self, configuration):
        self.configuration = configuration
        self.cache = {}

    def characteristics(self):
        """
        Reads the model characteristics.
        """
        if "characteristics" in self.cache:
            return self.cache["characteristics"]
        result = {}
        source = self.configuration["characteristics"]
        logging.info(
            f"Reading model characteristics from '{source}'."
        )
        with tqdm(total=sum(1 for line in open(source)) - 1) as counter:
            with open(source) as data:
                data.readline()  # skip the title line
                reader = csv.reader(data)
                for row in reader:
                    entry = {}
                    for i, characteristic in enumerate(CHARACTERISTICS):
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
                    result[entry["Id"]] = entry
                    counter.update(1)
        self.cache["characteristics"] = result
        return result

    def results(self):
        """
        Reads the results of the model checking contest.
        """
        if "results" in self.cache:
            return self.cache["results"]
        result = []
        source = self.configuration["results"]
        characteristics = self.cache["characteristics"]
        logging.info(
            f"Reading mcc results from '{source}'."
        )
        with tqdm(total=sum(1 for line in open(source)) - 1) as counter:
            with open(source) as data:
                data.readline()  # skip the title line
                reader = csv.reader(data)
                for row in reader:
                    entry = {}
                    for i, rentry in enumerate(RESULTS):
                        entry[rentry] = value_of(row[i])
                    if entry["Time OK"] \
                            and entry["Memory OK"] \
                            and entry["Status"] == "normal" \
                            and entry["Results"] not in ["DNC", "DNF", "CC"]:
                        for technique in re.findall(
                                r"([A-Z_]+)",
                                entry["Techniques"]
                        ):
                            if technique not in TECHNIQUES:
                                TECHNIQUES.append(technique)
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
                        if entry["Model Id"] in characteristics:
                            model = characteristics[entry["Model Id"]]
                            entry["Model"] = model
                            for key in model.keys():
                                if key != "Id":
                                    entry[key] = model[key]
                        entry["Time"] = entry["Clock Time"]
                        del entry["Time OK"]
                        del entry["Memory OK"]
                        del entry["CPU Time"]
                        del entry["Clock Time"]
                        del entry["IO Time"]
                        del entry["Cores"]
                        del entry["Results"]
                        del entry["Status"]
                        del entry["Techniques"]
                        result.append(entry)
                    counter.update(1)
        # Set techniques to False if they do not appear within an entry:
        with tqdm(total=len(result)) as counter:
            for entry in result:
                for technique in TECHNIQUES:
                    if technique not in entry:
                        entry[technique] = False
                counter.update(1)
        # Rename tools that are duplicated:
        with tqdm(total=len(result)) as counter:
            for entry in result:
                name = entry["Tool"]
                if name in self.configuration["renaming"]:
                    entry["Tool"] = self.configuration["renaming"][name]
                counter.update(1)
        self.cache["results"] = result
        return result

    def filter(self, predicate):
        """
        Filter the results given a predicate.
        """
        self.results()
        self.cache["results"] = [
            x for x in self.cache["results"]
            if predicate(x)
        ]
