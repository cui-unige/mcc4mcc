#! /usr/bin/env python3

"""
Model Checker Collection for the Model Checking Contest.
"""

import argparse
import logging
import os
import getpass
import json
import pathlib
import pickle
import platform
import re
import sys
import tempfile
import tarfile
import pandas
import xmltodict
import docker

from mcc.analysis import known, learned, score_of, max_score
from mcc.model import Values, Data, value_of


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

RENAMING = {
    "tapaalPAR": "tapaal",
    "tapaalSEQ": "tapaal",
    "tapaalEXP": "tapaal",
    "sift": "tina",
    "tedd": "tina",
}


temporary = None


def unarchive(filename):
    """
    Extract the model from an archive.
    """
    global temporary
    while True:
        if os.path.isfile(filename):
            directory = tempfile.TemporaryDirectory()
            temporary = directory
            logging.info(
                f"Extracting archive '{filename}' "
                f"to temporary directory '{directory.name}'.")
            with tarfile.open(name=filename) as tar:
                tar.extractall(path=directory.name)
            if platform.system() == "Darwin":
                filename = "/private" + directory.name
            else:
                filename = directory.name
        elif os.path.isdir(filename):
            if os.path.isfile(filename + "/model.pnml"):
                logging.info(
                    f"Using directory '{filename}' for input, "
                    f"as it contains a 'model.pnml' file.")
                break
            else:
                parts = os.listdir(filename)
                if len(parts) == 1:
                    filename = filename + "/" + parts[0]
                else:
                    logging.error(
                        f"Cannot use directory '{filename}' for input, "
                        f"as it does not contain a 'model.pnml' file.")
                    return None
        else:
            logging.error(
                f"Cannot use directory '{filename}' for input, "
                f"as it does not contain a 'model.pnml' file.")
            return None
    return filename


def read_boolean(filename):
    """Read a Boolean file from the MCC."""
    with open(filename, "r") as boolfile:
        what = boolfile.readline().strip()
        return value_of(what)


def do_extract(arguments):
    """
    Main function for the extract command.
    """
    data = Data({
        "characteristics": arguments.characteristics,
        "results": arguments.results,
        "renaming": RENAMING,
    })
    # Read data:
    data.characteristics()
    data.results()
    examinations = {x["Examination"] for x in data.results()}
    tools = {x["Tool"] for x in data.results()}
    # Filter by year:
    if arguments.year is not None:
        logging.info(
            f"Filtering year '{arguments.year}'."
        )
        data.filter(lambda e: e["Year"] == arguments.year)
    # Compute maximum score:
    logging.info(f"Maximum score is {max_score(data)}.")
    # Extract known data:
    known_data = known(data)
    with open(f"{arguments.data}/known.json", "w") as output:
        json.dump(known_data, output)
    # Extract learned data:
    learned_data, values = learned(data, {
        "Duplicates": arguments.duplicates,
        "Output Trees": arguments.output_trees,
    })
    with open(f"{arguments.data}/learned.json", "w") as output:
        json.dump(learned_data, output)
    with open(f"{arguments.data}/values.json", "w") as output:
        json.dump(values.items, output)
    # Compute scores for tools:
    for tool in tools:
        logging.info(f"Computing score of tool: '{tool}'.")
        score = score_of(data, tool)
        subresult = {
            "Algorithm": tool,
        }
        total = 0
        for key, value in score.items():
            subresult[key] = value
            total = total + value
        learned_data.append(subresult)
        logging.info(f"  Score: {total}")
    # Output per-examination scores:
    srt = []
    for subresult in learned_data:
        for examination in examinations:
            srt.append({
                "Name": subresult["Algorithm"],
                "Examination": examination,
                "Score": subresult[examination],
            })
    srt = sorted(srt, key=lambda e: (
        e["Examination"], e["Score"], e["Name"]
    ), reverse=True)
    for element in srt:
        examination = element["Examination"]
        score = element["Score"]
        name = element["Name"]
        logging.info(f"In {examination} : {score} for {name}.")


def do_run(arguments):
    """
    Main function for the run command.
    """
    # Load known info:
    logging.info(
        f"Reading known information in '{arguments.data}/known.json'."
    )
    with open(f"{arguments.data}/known.json", "r") as i:
        known_data = json.load(i)
    # Load learned info:
    logging.info(
        f"Reading learned information in '{arguments.data}/learned.json'."
    )
    with open(f"{arguments.data}/learned.json", "r") as i:
        learned_data = json.load(i)
    # Load translations:
    logging.info(
        f"Reading value translations in '{arguments.data}/values.json'."
    )
    with open(f"{arguments.data}/values.json", "r") as i:
        translations = json.load(i)
    values = Values(translations)
    # Find input:
    directory = unarchive(arguments.input)
    last = pathlib.PurePath(directory).stem
    split = re.search(r"([^-]+)\-([^-]+)\-([^-]+)$", last)
    if split is None:
        instance = last
        model = last
    else:
        instance = last
        model = split.group(1)
    logging.info(f"Using '{instance}' as instance name.")
    logging.info(f"Using '{model}' as model name.")
    examination = arguments.examination
    # Set tool:
    known_tools = None
    if arguments.tool is not None:
        known_tools = [arguments.tool]
    else:
        # Find known tools:
        if known_data[examination] is not None:
            if known_data[examination][instance] is not None:
                known_tools = known_data[examination][instance]
            elif known_data[examination][model] is not None:
                known_tools = known_data[examination][model]
    if known_tools is None:
        logging.warning(
            f"Cannot find known information for examination '{examination}' "
            f"on instance '{instance}' or model '{model}'.")
    # Set algorithm:
    learned_tools = None
    if arguments.algorithm:
        algorithm = arguments.algorithm
    else:
        algorithm = sorted(
            learned_data,
            key=lambda e: e[arguments.examination],
            reverse=True,
        )[0]["Algorithm"]
    logging.info(f"Using machine learning algorithm '{algorithm}'.")
    with open(f"{arguments.data}/learned.{algorithm}.p", "rb") as i:
        model = pickle.load(i)
    # Find learned tools:
    is_colored = read_boolean(f"{directory}/iscolored")
    if is_colored:
        has_pt = read_boolean(f"{directory}/equiv_pt")
    else:
        has_colored = read_boolean(f"{directory}/equiv_col")
    with open(f"{directory}/GenericPropertiesVerdict.xml", "r") as i:
        verdict = xmltodict.parse(i.read())
    characteristics = {
        "Examination": examination,
        "Place/Transition": (not is_colored) or has_pt,
        "Colored": is_colored or has_colored,
    }
    for value in verdict["toolspecific"]["verdict"]:
        if value["@value"] == "true":
            characteristics[VERDICTS[value["@reference"]]] = True
        elif value["@value"] == "false":
            characteristics[VERDICTS[value["@reference"]]] = False
        else:
            characteristics[VERDICTS[value["@reference"]]] = None
    logging.info(f"Model characteristics are: {characteristics}.")
    # Load characteristics for machine learning:
    test = {}
    for key, value in characteristics.items():
        test[key] = values.to_learning(value)
    # http://scikit-learn.org/stable/modules/model_persistence.html
    predicted = model.predict(pandas.DataFrame([test]))
    learned_tools = [{"Tool": values.from_learning(predicted[0])}]
    logging.info(f"Known tools are: {known_tools}.")
    logging.info(f"Learned tools are: {learned_tools}.")
    # Evaluate quality of learned tool:
    if known_tools is not None and learned_tools is not None:
        found = learned_tools[0]
        best = known_tools[0]
        distance = None
        for entry in known_tools:
            if entry["Tool"] == found["Tool"]:
                found_tool = entry["Tool"]
                best_tool = best["Tool"]
                distance = entry["Time"] / best["Time"]
                logging.info(f"Learned tool {found_tool} is {distance}x "
                             f"far from the best tool {best_tool}.")
                break
        if distance is None:
            logging.info(f"Learned tool does not appear within known.")
    elif known_tools is None:
        logging.warning(f"No known information "
                        f"for examination '{examination}' "
                        f"on instance '{instance}' or model '{model}'.")
    elif learned_tools is None:
        logging.warning(f"No learned information "
                        f"for examination '{examination}' "
                        f"on instance '{instance}' or model '{model}'.")
    # Run the tools:
    if known_tools is not None:
        tools = known_tools
    elif learned_tools is not None:
        tools = learned_tools
    else:
        logging.error(f"DO NOT COMPETE")
        sys.exit(1)
    path = os.path.abspath(directory)
    # Load docker client:
    client = docker.from_env()
    for entry in tools:
        try:
            tool = entry["Tool"]
            logging.info(f"{examination} {tool} {instance}...")
            # client.images.pull(f"mccpetrinets/{tool.lower()}")
            logs = client.containers.run(
                image=f"mccpetrinets/{tool.lower()}",
                entrypoint="mcc-head",
                command=[],
                auto_remove=True,
                stdout=True,
                stderr=True,
                detach=False,
                working_dir="/mcc-data",
                volumes={
                    f"{path}": {
                        "bind": "/mcc-data",
                        "mode": "rw",
                    },
                },
                environment={
                    "BK_LOG_FILE": "/mcc-data/log",
                    "BK_EXAMINATION": f"{examination}",
                    "BK_TIME_CONFINEMENT": "3600",
                    "BK_INPUT": f"{instance}",
                    "BK_TOOL": tool.lower(),
                },
            )
            logging.info(logs)
            sys.exit(0)
        except docker.errors.ContainerError as error:
            logging.error(f"  Failure", error)
        except docker.errors.ImageNotFound as error:
            logging.error(f"  Unexpected error", error)
        except docker.errors.APIError as error:
            logging.error(f"  Unexpected error", error)
    logging.error(f"CANNOT COMPUTE")
    sys.exit(1)


def do_test(arguments):
    """
    Main function for the test command.
    """
    data = Data({
        "characteristics": arguments.characteristics,
        "results": arguments.results,
        "renaming": RENAMING,
    })
    # Read data:
    data.characteristics()
    data.results()
    # Filter by year:
    if arguments.year is not None:
        logging.info(
            f"Filtering year '{arguments.year}'."
        )
        data.filter(lambda e: e["Year"] == arguments.year)
    results = data.results()
    examinations = {x["Examination"] for x in results}
    tools = {x["Tool"] for x in results}
    # Use arguments:
    path = arguments.models
    if arguments.tool is not None:
        tools = [arguments.tool]
    # Load docker client:
    client = docker.from_env()
    client.login(
        username=input("Docker username: "),
        password=getpass.getpass("Docker password: "),
    )
    tested = {}
    for examination in examinations:
        tested[examination] = {}
        for tool in tools:
            srt = sorted([
                entry for entry in results
                if entry["Examination"] == examination
                and entry["Tool"] == tool
            ], key=lambda e: e["Time"])
            if not srt:
                logging.info(f"No test available for {examination} {tool}.")
                continue
            instance = srt[0]["Instance"]
            directory = unarchive(f"{path}/{instance}.tgz")
            logging.info(f"Testing {examination} {tool} with {instance}...")
            try:
                # client.images.pull(f"mccpetrinets/{tool.lower()}")
                logs = client.containers.run(
                    image=f"mccpetrinets/{tool.lower()}",
                    entrypoint="mcc-head",
                    command=[],
                    auto_remove=True,
                    stdout=True,
                    stderr=True,
                    detach=False,
                    working_dir="/mcc-data",
                    volumes={
                        f"{directory}": {
                            "bind": "/mcc-data",
                            "mode": "rw",
                        },
                    },
                    environment={
                        "BK_LOG_FILE": "/mcc-data/log",
                        "BK_examination": f"{examination}",
                        "BK_TIME_CONFINEMENT": "3600",
                        "BK_INPUT": f"{instance}",
                        "BK_TOOL": tool.lower(),
                    },
                )
                logging.info(logs)
                tested[examination][tool] = True
            except docker.errors.ContainerError as error:
                logging.error(f"  Failure", error)
                tested[examination][tool] = False
            except docker.errors.ImageNotFound as error:
                logging.error(f"  Unexpected error", error)
                tested[examination][tool] = False
            except docker.errors.APIError as error:
                logging.error(f"  Unexpected error", error)
                tested[examination][tool] = False
    for examination, subresults in tested.items():
        logging.error(f"Tests for {examination}:")
        for tool, value in subresults.items():
            logging.error(f"  {tool}: {value}")


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

PARSER = argparse.ArgumentParser(
    "model checker collection for the model checking contest"
)
PARSER.add_argument(
    "--data",
    help="directory containing data known and learned from models",
    type=str,
    dest="data",
    default=os.getcwd(),
)

SUBPARSERS = PARSER.add_subparsers()
EXTRACT = SUBPARSERS.add_parser(
    "extract",
    description="Data extractor",
)
EXTRACT.add_argument(
    "--results",
    help="results of the model checking contest",
    type=str,
    dest="results",
    default=os.getcwd() + "/results.csv",
)
EXTRACT.add_argument(
    "--characteristics",
    help="model characteristics from the Petri net repository",
    type=str,
    dest="characteristics",
    default=os.getcwd() + "/characteristics.csv",
)
EXTRACT.add_argument(
    "--year",
    help="Use results for a specific year (YYYY format).",
    type=int,
    dest="year",
)
EXTRACT.add_argument(
    "--duplicates",
    help="Allow duplicate entries",
    type=bool,
    dest="duplicates",
    default=False,
)
EXTRACT.add_argument(
    "--output-trees",
    help="Output decision trees",
    type=bool,
    dest="output_trees",
    default=False,
)
EXTRACT.set_defaults(func=do_extract)

TEST = SUBPARSERS.add_parser(
    "test",
    description="Test",
)
TEST.add_argument(
    "--results",
    help="results of the model checking contest",
    type=str,
    dest="results",
    default=os.getcwd() + "/results.csv",
)
TEST.add_argument(
    "--characteristics",
    help="model characteristics from the Petri net repository",
    type=str,
    dest="characteristics",
    default=os.getcwd() + "/characteristics.csv",
)
TEST.add_argument(
    "--year",
    help="Use results for a specific year (YYYY format).",
    type=int,
    dest="year",
)
TEST.add_argument(
    "--models",
    help="directory containing all models",
    type=str,
    dest="models",
    default=os.getcwd() + "/models",
)
TEST.add_argument(
    "--tool",
    help="only tool to test",
    type=str,
    dest="tool",
)
TEST.set_defaults(func=do_test)

RUN = SUBPARSERS.add_parser(
    "run",
    description="Runner",
)
RUN.add_argument(
    "--input",
    help="input directory or archive",
    type=str,
    dest="input",
    default=os.getcwd(),
)
RUN.add_argument(
    "--examination",
    help="examination type",
    type=str,
    dest="examination",
    default=os.getenv("BK_EXAMINATION"),
)
RUN.add_argument(
    "--tool",
    help="tool to use",
    type=str,
    dest="tool",
)
RUN.add_argument(
    "--algorithm",
    help="machine learning algorithm to use",
    type=str,
    dest="algorithm",
)
RUN.set_defaults(func=do_run)

ARGUMENTS = PARSER.parse_args()
if "func" in ARGUMENTS:
    ARGUMENTS.func(ARGUMENTS)
else:
    PARSER.print_usage()
