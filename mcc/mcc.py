#! /usr/bin/env python3

"""
Run the Model Checking Contest tool.
"""

if __name__ == "__main__":

    import argparse
    import json
    import logging
    import os
    import pathlib
    import pickle
    import re
    import subprocess
    import sys
    import tempfile
    import tarfile
    import pandas
    import xmltodict

    import extract

    def knn_distance(lhs, rhs, bound=2):
        """The knn distance function, required to load knn algorithms."""
        return extract.knn_distance(lhs, rhs, bound)

    def read_boolean(filename):
        """Read a Boolean file from the MCC."""
        with open(filename, "r") as boolfile:
            what = boolfile.readline().strip()
            return extract.value_of(what)

    PARSER = argparse.ArgumentParser(
        description="Model Checker Collection for the Model Checking Contest"
    )
    PARSER.add_argument(
        "--input",
        help="input directory or archive",
        type=str,
        dest="input",
        default=os.getcwd(),
    )
    PARSER.add_argument(
        "--data",
        help="directory containing data known and learned from models",
        type=str,
        dest="data",
        default=os.getcwd(),
    )
    PARSER.add_argument(
        "--algorithm",
        help="machine learning algorithm to use",
        type=str,
        dest="algorithm",
    )
    PARSER.add_argument(
        "--examination",
        help="examination type",
        type=str,
        dest="examination",
        default=os.getenv("BK_EXAMINATION"),
    )
    PARSER.add_argument(
        "--tool",
        help="tool",
        type=str,
        dest="tool",
    )
    PARSER.add_argument(
        "--evaluate",
        help="compare the known and learned tool",
        type=bool,
        dest="evaluate",
        default=True,
    )
    ARGUMENTS = PARSER.parse_args()
    logging.basicConfig(level=logging.INFO)

    VERDICTS = {
        "ORDINARY": "Ordinary",
        "SIMPLE_FREE_CHOICE": "Simple Free Choice",
        "EXTENDED_FREE_CHOICE": "Extended Free Choice",
        "STATE_MACHINE": "State Machine",
        "MARKED_GRAPH": "Marked Graph",
        "CONNECTED": "Connected",
        "STRONGLY_CONNECTED": "Strongly Connected",
        "SOURCE_PLACE": "Source Place",
        "SINK_PLACE": "Sink Place",
        "SOURCE_TRANSITION": "Source Transition",
        "SINK_TRANSITION": "Sink Transition",
        "LOOP_FREE": "Loop Free",
        "CONSERVATIVE": "Conservative",
        "SUBCONSERVATIVE": "Sub-Conservative",
        "NESTED_UNITS": "Nested Units",
        "SAFE": "Safe",
        "DEADLOCK": "Deadlock",
        "REVERSIBLE": "Reversible",
        "QUASI_LIVE": "Quasi Live",
        "LIVE": "Live",
    }

    logging.info(
        f"Reading known information in '{ARGUMENTS.data}/known.json'.")
    with open(f"{ARGUMENTS.data}/known.json", "r") as i:
        KNOWN = json.load(i)

    logging.info(
        f"Reading learned information in '{ARGUMENTS.data}/learned.json'.")
    with open(f"{ARGUMENTS.data}/learned.json", "r") as i:
        LEARNED = json.load(i)
        extract.translate.ITEMS = LEARNED["translation"]

    if ARGUMENTS.algorithm is None:
        ALGORITHM = sorted(
            LEARNED["algorithms"],
            key=lambda e: e["mean"],
            reverse=True
        )[0]["algorithm"]
    else:
        ALGORITHM = ARGUMENTS.algorithm
    logging.info(f"Using machine learning algorithm '{ALGORITHM}'.")

    while True:
        if os.path.isfile(ARGUMENTS.input):
            DIRECTORY = tempfile.TemporaryDirectory()
            logging.info(
                f"Extracting archive '{ARGUMENTS.input}' "
                f"to temporary directory '{DIRECTORY}'.")
            with tarfile.open(name=ARGUMENTS.input) as tar:
                tar.extractall(path=DIRECTORY)
            ARGUMENTS.input = DIRECTORY
        elif os.path.isdir(ARGUMENTS.input):
            if os.path.isfile(ARGUMENTS.input + "/model.pnml"):
                logging.info(
                    f"Using directory '{ARGUMENTS.input}' for input, "
                    f"as it contains a 'model.pnml' file.")
                break
            else:
                logging.error(
                    f"Cannot use directory '{ARGUMENTS.input}' for input, "
                    f"as it does not contain a 'model.pnml' file.")
                sys.exit(1)
        else:
            logging.error(
                f"Cannot use directory '{ARGUMENTS.input}' for input, "
                f"as it does not contain a 'model.pnml' file.")
            sys.exit(1)

    LAST = pathlib.PurePath(ARGUMENTS.input).stem
    SPLIT = re.search(r"([^-]+)\-([^-]+)\-([^-]+)$", LAST)
    if SPLIT is None:
        INSTANCE = LAST
        MODEL = LAST
    else:
        INSTANCE = LAST
        MODEL = SPLIT.group(1)
    logging.info(f"Using '{INSTANCE}' as instance name.")
    logging.info(f"Using '{MODEL}' as model name.")

    EXAMINATION = ARGUMENTS.examination
    if ARGUMENTS.tool is not None:
        logging.info(f"Using only the tool '{ARGUMENTS.tool}'.")
        USER_TOOLS = [ARGUMENTS.tool]
    else:
        USER_TOOLS = None
    if EXAMINATION in KNOWN \
            and MODEL in KNOWN[EXAMINATION] \
            and INSTANCE in KNOWN[EXAMINATION][MODEL]:
        KNOWN_TOOLS = KNOWN[EXAMINATION][MODEL][INSTANCE]["sorted"]
    elif EXAMINATION in KNOWN \
            and MODEL in KNOWN[EXAMINATION]:
        KNOWN_TOOLS = KNOWN[EXAMINATION][MODEL]["sorted"]
    else:
        KNOWN_TOOLS = None
        logging.warning(
            f"Cannot find known information for examination '{EXAMINATION}' "
            f"on instance '{INSTANCE}' or model '{MODEL}'.")

    LEARNED_TOOLS = None
    IS_COLORED = read_boolean(f"{ARGUMENTS.input}/iscolored")
    if IS_COLORED:
        HAS_PT = read_boolean(f"{ARGUMENTS.input}/equiv_pt")
    else:
        HAS_COLORED = read_boolean(f"{ARGUMENTS.input}/equiv_col")
    with open(f"{ARGUMENTS.input}/GenericPropertiesVerdict.xml", "r") as i:
        VERDICT = xmltodict.parse(i.read())
    CHARACTERISTICS = {
        "Examination": EXAMINATION,
        "Place/Transition": (not IS_COLORED) or HAS_PT,
        "Colored": IS_COLORED or HAS_COLORED,
    }
    for v in VERDICT["toolspecific"]["verdict"]:
        if v["@value"] == "true":
            CHARACTERISTICS[VERDICTS[v["@reference"]]] = True
        elif v["@value"] == "false":
            CHARACTERISTICS[VERDICTS[v["@reference"]]] = False
        else:
            CHARACTERISTICS[VERDICTS[v["@reference"]]] = None
    logging.info(f"Model characteristics are: {CHARACTERISTICS}.")
    with open(f"{ARGUMENTS.data}/learned.{ALGORITHM}.p", "rb") as i:
        MODEL = pickle.load(i)
    TEST = {}
    for key, value in CHARACTERISTICS.items():
        TEST[key] = extract.translate(value)
    # http://scikit-learn.org/stable/modules/model_persistence.html
    PREDICTED = MODEL.predict(pandas.DataFrame([TEST]))
    LEARNED_TOOLS = [{"tool": extract.translate_back(PREDICTED[0])}]

    if ARGUMENTS.evaluate:
        logging.info(f"Known tools are: {KNOWN_TOOLS}.")
        logging.info(f"Learned tool is: {LEARNED_TOOLS[0]}.")
        if KNOWN_TOOLS is not None and LEARNED_TOOLS is not None:
            LEARNED = LEARNED_TOOLS[0]
            BEST = KNOWN_TOOLS[0]
            DISTANCE = None
            for entry in KNOWN_TOOLS:
                if entry["tool"] == LEARNED["tool"]:
                    learned_tool = entry["tool"]
                    best_tool = BEST["tool"]
                    DISTANCE = entry["time"] / BEST["time"]
                    logging.info(f"Learned tool {learned_tool} is {DISTANCE} "
                                 f"far from the best tool {best_tool}.")
                    break
            if DISTANCE is None:
                logging.info(f"Learned tool does not appear within known.")
        elif KNOWN_TOOLS is None:
            logging.warning(f"No known information "
                            f"for examination '{EXAMINATION}' "
                            f"on instance '{INSTANCE}' or model '{MODEL}'.")
        elif LEARNED_TOOLS is None:
            logging.warning(f"No learned information "
                            f"for examination '{EXAMINATION}' "
                            f"on instance '{INSTANCE}' or model '{MODEL}'.")

    if USER_TOOLS is not None:
        TOOLS = USER_TOOLS
    if KNOWN_TOOLS is not None:
        TOOLS = KNOWN_TOOLS
    elif LEARNED_TOOLS is not None:
        TOOLS = LEARNED_TOOLS
    else:
        logging.error(f"DO NOT COMPETE")
        sys.exit(1)

    SUCCESS = None
    PATH = os.path.abspath(ARGUMENTS.input)
    for x in TOOLS:
        tool = x["tool"]
        logging.info(f"Starting tool '{tool}'...")
        command = [
            "docker",
            "run",
            "--volume", f"{PATH}:/mcc-data",
            "--workdir", "/mcc-data",
        ]
        for key, value in os.environ.items():
            if key.startswith("BK_"):
                command.append("--env")
                command.append(f"{key}={value}")
        command.append(f"mcc/{tool}".lower())
        logging.info(f"Running {command}.")
        SUCCESS = subprocess.call(command)
        if SUCCESS == 0:
            break
    if SUCCESS != 0:
        logging.error(f"CANNOT COMPUTE")
        sys.exit(1)
