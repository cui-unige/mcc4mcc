#! /usr/bin/env python3

"""
Model Checker Collection for the Model Checking Contest.
"""

import argparse
import logging
import os
# import getpass
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

from mcc.analysis import known, learned, score_of, max_score, \
    characteristics_of
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

TEMPORARY = None


def unarchive(filename):
    """
    Extract the model from an archive.
    """
    # pylint: disable=global-statement
    global TEMPORARY
    # pylint: enable=global-statement
    while True:
        if os.path.isfile(filename):
            directory = tempfile.TemporaryDirectory()
            TEMPORARY = directory
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
    if arguments.forget is None:
        arguments.forget = []
    else:
        arguments.forget = arguments.forget.split(",")
    data = Data({
        "characteristics": arguments.characteristics,
        "results": arguments.results,
        "renaming": RENAMING,
        "forget": arguments.forget,
    })
    # Read data:
    data.characteristics()
    # Compute the characteristics for models:
    characteristics_of(data)
    # Use results:
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
    with open(f"{arguments.data}/values.json", "w") as output:
        json.dump(values.items, output)
    # Compute scores for tools:
    for tool in tools:
        logging.info(f"Computing score of tool: '{tool}'.")
        score = score_of(data, tool)
        subresult = {
            "Algorithm": tool,
            "Is-Tool": True,
            "Is-Algorithm": False,
        }
        total = 0
        for key, value in score.items():
            subresult[key] = value
            total = total + value
        learned_data.append(subresult)
        logging.info(f"  Score: {total}")
    with open(f"{arguments.data}/learned.json", "w") as output:
        json.dump(learned_data, output)
    # Print per-examination scores:
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
        if score > 0:
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
    if arguments.instance is None:
        instance = pathlib.PurePath(directory).stem
    else:
        instance = arguments.instance
    split = re.search(r"([^-]+)\-([^-]+)\-([^-]+)$", instance)
    model = split.group(1)
    logging.info(f"Using '{instance}' as instance name.")
    logging.info(f"Using '{model}' as model name.")
    examination = arguments.examination
    # Set tool:
    known_tools = None
    if arguments.tool is not None:
        known_tools = [{
            "Tool": arguments.tool,
            "Time": None,
            "Memory": None,
        }]
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
        algorithm = sorted(
            [x for x in learned_data
             if x["Algorithm"] == arguments.algorithm],
            key=lambda e: e[arguments.examination],
            reverse=True,
        )[0]["Algorithm"]
        with open(f"{arguments.data}/learned.{algorithm}.p", "rb") as i:
            model = pickle.load(i)
    elif arguments.cheat:
        algorithm = sorted(
            [x for x in learned_data if x["Is-Tool"]],
            key=lambda e: e[arguments.examination],
            reverse=True,
        )[0]["Algorithm"]

        # pylint: disable=missing-docstring,no-self-use
        class Silly:
            def predict(self, _):
                return [values.to_learning(algorithm)]
        # pylint: enable=missing-docstring,no-self-use

        model = Silly()
    else:
        algorithm = sorted(
            [x for x in learned_data if x["Is-Algorithm"]],
            key=lambda e: e[arguments.examination],
            reverse=True,
        )[0]["Algorithm"]
        with open(f"{arguments.data}/learned.{algorithm}.p", "rb") as i:
            model = pickle.load(i)
    logging.info(f"Using algorithm or tool '{algorithm}'.")
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
        if arguments.tool is None:
            distance = None
            for entry in known_tools:
                if entry["Tool"] == found["Tool"]:
                    found_tool = entry["Tool"]
                    best_tool = best["Tool"]
                    distance = entry["Time"] / best["Time"]
                    logging.info(f"Learned tool {found_tool} is {distance}x "
                                 f"far from the best tool {best_tool}.")
                    break
    elif known_tools is None:
        logging.warning(f"No known information "
                        f"for examination '{examination}' "
                        f"on instance '{instance}' or model '{model}'.")
    elif learned_tools is None:
        logging.warning(f"No learned information "
                        f"for examination '{examination}' "
                        f"on instance '{instance}' or model '{model}'.")
    # Run the tools:
    if arguments.cheat and known_tools is not None:
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
        tool = entry["Tool"]
        logging.info(f"{examination} {tool} {instance}...")
        # client.images.pull(f"mccpetrinets/{tool.lower()}")
        container = client.containers.run(
            image=f"mccpetrinets/{tool.lower()}",
            command="mcc-head",
            auto_remove=False,
            stdout=True,
            stderr=True,
            detach=True,
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
        result = container.wait()
        for line in container.logs(stream=True):
            logging.info(line.decode("UTF-8").strip())
        container.remove()
        if result["StatusCode"] == 0:
            sys.exit(0)
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
    if arguments.examination is not None:
        examinations = [arguments.examination]
    if arguments.tool is not None:
        tools = [arguments.tool]
    # Load docker client:
    client = docker.from_env()
    tested = {}
    for examination in examinations:
        tested[examination] = {}
        for tool in tools:
            instances = sorted([
                entry for entry in results
                if entry["Examination"] == examination
                and entry["Tool"] == tool
            ], key=lambda e: e["Time"])
            if arguments.instance:
                instances = [{
                    "Examination": examination,
                    "Tool": tool,
                    "Instance": arguments.instance,
                }]
            if instances:
                instance = instances[0]["Instance"]
                logging.info("")
                logging.info("==============================================")
                logging.info(f"Testing {examination} {tool} with {instance}.")
                logging.info("==============================================")
                logging.info("")
            else:
                logging.info("")
                logging.info("==============================================")
                logging.warning(f"No test for {examination} {tool}.")
                logging.info("==============================================")
                logging.info("")
                continue
            directory = unarchive(f"{path}/{instance}.tgz")
            try:
                container = client.containers.run(
                    image=f"mccpetrinets/{tool.lower()}",
                    command="mcc-head",
                    auto_remove=False,
                    stdout=True,
                    stderr=True,
                    detach=True,
                    working_dir="/mcc-data",
                    volumes={
                        f"{directory}": {
                            "bind": "/mcc-data",
                            "mode": "rw",
                        },
                    },
                    environment={
                        "BK_LOG_FILE": "/mcc-data/log",
                        "BK_EXAMINATION": f"{examination}",
                        "BK_TIME_CONFINEMENT": "3600",
                        "BK_INPUT": f"{instance}",
                        "BK_TOOL": tool,
                    },
                )
                result = container.wait()
                print(result)
                # if result["StatusCode"] == 0:
                #     tested[examination][tool] = True
                # else:
                tested[examination][tool] = result["StatusCode"] == 0
                for line in container.logs(stream=True):
                    logging.info(line.decode("UTF-8").strip())
                container.remove()
            except docker.errors.NotFound:
                logging.warning(f"Docker image does not exist.")
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
    dest="duplicates",
    action="store_true"
)
EXTRACT.add_argument(
    "--output-trees",
    help="Output decision trees",
    type=bool,
    dest="output_trees",
    default=False,
)
EXTRACT.add_argument(
    "--forget",
    help="Forget characteristics (comma separated)",
    dest="forget",
    default=None,
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
TEST.add_argument(
    "--examination",
    help="examination type",
    type=str,
    dest="examination",
)
TEST.add_argument(
    "--instance",
    help="instance",
    type=str,
    dest="instance",
)
TEST.set_defaults(func=do_test)

RUN = SUBPARSERS.add_parser(
    "run",
    description="Runner",
)
RUN.add_argument(
    "--input",
    help="archive or directory containing the model",
    type=str,
    dest="input",
    default=os.getcwd(),
)
RUN.add_argument(
    "--instance",
    help="instance name",
    type=str,
    dest="instance",
    default=os.getenv("BK_INPUT"),
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
RUN.add_argument(
    "--cheat",
    help="Cheat by using known information",
    dest="cheat",
    action="store_true",
    default=os.getenv("BK_TOOL") == "mcc4mcc-cheat"
)
RUN.set_defaults(func=do_run)

ARGUMENTS = PARSER.parse_args()
if "func" in ARGUMENTS:
    ARGUMENTS.func(ARGUMENTS)
else:
    PARSER.print_usage()
