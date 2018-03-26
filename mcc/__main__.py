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
import re
import readline
import subprocess
import sys
import tempfile
import tarfile
import pandas
import xmltodict
import docker

import mcc.analysis
import mcc.model


def unarchive(filename):
    """
    Extract the model from an archive.
    """
    while True:
        if os.path.isfile(filename):
            directory = tempfile.TemporaryDirectory()
            logging.info(
                f"Extracting archive '{filename}' "
                f"to temporary directory '{directory}'.")
            with tarfile.open(name=filename) as tar:
                tar.extractall(path=directory.name)
            filename = directory.name
        elif os.path.isdir(filename):
            if os.path.isfile(filename + "/model.pnml"):
                logging.info(
                    f"Using directory '{filename}' for input, "
                    f"as it contains a 'model.pnml' file.")
                break
            else:
                logging.error(
                    f"Cannot use directory '{filename}' for input, "
                    f"as it does not contain a 'model.pnml' file.")
                sys.exit(1)
        else:
            logging.error(
                f"Cannot use directory '{filename}' for input, "
                f"as it does not contain a 'model.pnml' file.")
            sys.exit(1)
    return filename


def do_extract(arguments):
    """
    Main function for the extract command.
    """
    data = mcc.model.Data({
        "characteristics": arguments.characteristics,
        "results": arguments.results,
        "renaming": {
            "tapaalPAR": "tapaal",
            "tapaalSEQ": "tapaal",
            "tapaalEXP": "tapaal",
            "sift": "tina",
            "tedd": "tina",
        },
    })
    logging.info(
        f"Reading model characteristics from '{arguments.characteristics}'."
    )
    data.characteristics()
    logging.info(
        f"Reading mcc results from '{arguments.results}'."
    )
    data.results()
    if arguments.year is not None:
        logging.info(
            f"Filtering year '{arguments.year}'."
        )
        data.filter(lambda e: e["Year"] == arguments.year)
    logging.info(
        f"Analyzing known data."
    )
    known = mcc.analysis.known(data)
    with open(f"{arguments.data}/known.json", "w") as output:
        json.dump(known, output)


def do_run(arguments):
    """
    Main function for the run command.
    """
    verdicts = {
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
        f"Reading known information in '{arguments.data}/known.json'."
    )
    with open(f"{arguments.data}/known.json", "r") as i:
        known = json.load(i)
    # logging.info(
    #     f"Reading learned information in '{arguments.data}/learned.json'."
    # )
    # with open(f"{arguments.data}/learned.json", "r") as i:
    #     learned = json.load(i)
    #     # FIXME
    #     # extract.translate.ITEMS = learned["translation"]
    arguments.input = unarchive(arguments.input)
    last = pathlib.PurePath(arguments.input).stem
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
    known_tools = None
    if known[examination] is not None:
        if known[examination][instance] is not None:
            known_tools = known[examination][instance]
        elif known[examination][model] is not None:
            known_tools = known[examination][model]
    if known_tools is None:
        logging.warning(
            f"Cannot find known information for examination '{examination}' "
            f"on instance '{instance}' or model '{model}'.")
    learned_tools = None
    # IS_COLORED = read_boolean(f"{arguments.input}/iscolored")
    # if IS_COLORED:
    #     HAS_PT = read_boolean(f"{arguments.input}/equiv_pt")
    # else:
    #     HAS_COLORED = read_boolean(f"{arguments.input}/equiv_col")
    # with open(f"{arguments.input}/GenericPropertiesVerdict.xml", "r") as i:
    #     VERDICT = xmltodict.parse(i.read())
    # CHARACTERISTICS = {
    #     "Examination": examination,
    #     "Place/Transition": (not IS_COLORED) or HAS_PT,
    #     "Colored": IS_COLORED or HAS_COLORED,
    # }
    # for v in VERDICT["toolspecific"]["verdict"]:
    #     if v["@value"] == "true":
    #         CHARACTERISTICS[verdicts[v["@reference"]]] = True
    #     elif v["@value"] == "false":
    #         CHARACTERISTICS[verdicts[v["@reference"]]] = False
    #     else:
    #         CHARACTERISTICS[verdicts[v["@reference"]]] = None
    # logging.info(f"Model characteristics are: {CHARACTERISTICS}.")
    # with open(f"{arguments.data}/learned.{ALGORITHM}.p", "rb") as i:
    #     model = pickle.load(i)
    # TEST = {}
    # for key, value in CHARACTERISTICS.items():
    #     TEST[key] = processing.translate(value)
    # # http://scikit-learn.org/stable/modules/model_persistence.html
    # PREDICTED = model.predict(pandas.DataFrame([TEST]))
    # learned_tools = [{"tool": processing.translate_back(PREDICTED[0])}]
    # logging.info(f"Known tools are: {known_tools}.")
    # logging.info(f"Learned tool is: {learned_tools[0]}.")
    # if known_tools is not None and learned_tools is not None:
    #     learned = learned_tools[0]
    #     BEST = known_tools[0]
    #     DISTANCE = None
    #     for entry in known_tools:
    #         if entry["tool"] == learned["tool"]:
    #             learned_tool = entry["tool"]
    #             best_tool = BEST["tool"]
    #             DISTANCE = entry["time"] / BEST["time"]
    #             logging.info(f"Learned tool {learned_tool} is {DISTANCE} "
    #                          f"far from the best tool {best_tool}.")
    #             break
    #     if DISTANCE is None:
    #         logging.info(f"Learned tool does not appear within known.")
    # elif known_tools is None:
    #     logging.warning(f"No known information "
    #                     f"for examination '{examination}' "
    #                     f"on instance '{instance}' or model '{model}'.")
    # elif learned_tools is None:
    #     logging.warning(f"No learned information "
    #                     f"for examination '{examination}' "
    #                     f"on instance '{instance}' or model '{model}'.")
    if known_tools is not None:
        tools = known_tools
    elif learned_tools is not None:
        tools = learned_tools
    else:
        logging.error(f"DO NOT COMPETE")
        sys.exit(1)
    # Run the tools:
    success = None
    path = os.path.abspath(arguments.input)
    # Load docker client:
    client = docker.from_env()
    # client.login(
    #     username=input("Docker username: "),
    #     password=getpass.getpass("Docker password: "),
    # )
    for entry in tools:
        try:
            tool = entry["Tool"]
            logging.info(f"{examination} {tool} {instance}...")
            # client.images.pull(f"mccpetrinets/{tool.lower()}")
            logs = client.containers.run(
                image=f"mccpetrinets/{tool.lower()}",
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
                    "BK_examination": f"{examination}",
                    "BK_TIME_CONFINEMENT": "3600",
                    "BK_INPUT": f"{instance}",
                    "BK_TOOL": tool.lower(),
                },
            )
            logging.info(logs)
            success = True
            break
        except docker.errors.ContainerError as error:
            success = False
            logging.error(f"  Failure", error)
        except docker.errors.ImageNotFound as error:
            success = False
            logging.error(f"  Unexpected error", error)
        except docker.errors.APIError as error:
            success = False
            logging.error(f"  Unexpected error", error)
    if success is None:
        logging.error(f"DO NOT COMPETE")
        sys.exit(1)
    if not success:
        logging.error(f"CANNOT COMPUTE")
        sys.exit(1)


def do_test(arguments):
    """
    Main function for the test command.
    """
    logging.info(
        f"Reading known information in '{arguments.data}/known.json'.")
    with open(f"{arguments.data}/known.json", "r") as i:
        known = json.load(i)
    # Load docker client:
    client = docker.from_env()
    client.login(
        username=input("Docker username: "),
        password=getpass.getpass("Docker password: "),
    )
    #
    path = os.path.abspath(arguments.models)
    tested = {}
    test_tool = arguments.tool
    for examination, what in known.items():
        if examination not in tested:
            tested[examination] = {}
        for instance, entries in what.items():
            for entry in entries:
                tool = entry["Tool"]
                if entry["Count"] \
                        or tool in tested[examination] \
                        or (test_tool is not None and tool != test_tool):
                    continue
                directory = unarchive(f"{path}/{instance}.tgz")
                try:
                    logging.info(f"{examination} {tool} {instance}...")
                    # client.images.pull(f"mccpetrinets/{tool.lower()}")
                    logs = client.containers.run(
                        image=f"mccpetrinets/{tool.lower()}",
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
EXTRACT.set_defaults(func=do_extract)

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
    default=os.getenv("BK_examination"),
)
RUN.set_defaults(func=do_run)

TEST = SUBPARSERS.add_parser(
    "test",
    description="Test",
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

# PARSER.add_argument(
#     "--learned",
#     help="data learned from models",
#     type=str,
#     dest="learned",
#     default=os.getcwd() + "/learned.json",
# )
# PARSER.add_argument(
#     "--distance",
#     help="Allowed distance from the best tool (in percent)",
#     type=float,
#     dest="distance",
# )
# PARSER.add_argument(
#     "--duplicates",
#     help="Allow duplicate entries",
#     type=bool,
#     dest="duplicates",
#     default=False,
# )
# PARSER.add_argument(
#     "--compute-score",
#     help="Compute score in the Model Checking Contest",
#     type=bool,
#     dest="mcc_score",
#     default=True,
# )
# PARSER.add_argument(
#     "--compute-distance",
#     help="Compute distance in the Model Checking Contest",
#     type=bool,
#     dest="mcc_distance",
#     default=True,
# )
# PARSER.add_argument(
#     "--useless",
#     help="Compute useless characteristics",
#     type=bool,
#     dest="useless",
#     default=False,
# )
# PARSER.add_argument(
#     "--output-dt",
#     help="Output the graph of trained decision tree.",
#     type=bool,
#     dest="output_dt",
#     default=False,
# )
ARGUMENTS = PARSER.parse_args()
ARGUMENTS.func(ARGUMENTS)
