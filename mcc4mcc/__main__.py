#! /usr/bin/env python3

"""
Model Checker Collection for the Model Checking Contest.
"""

import argparse
import hashlib
import math
import logging
import os
import random
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

from mcc4mcc.analysis import known, learned, score_of, max_score, \
    characteristics_of, REMOVE
from mcc4mcc.model import Values, Data, value_of, CHARACTERISTICS


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
                f"Extracting archive {filename} "
                f"to temporary directory {directory.name}.")
            with tarfile.open(name=filename) as tar:
                tar.extractall(path=directory.name)
            if platform.system() == "Darwin":
                filename = "/private" + directory.name
            else:
                filename = directory.name
        elif os.path.isdir(filename):
            if os.path.isfile(filename + "/model.pnml"):
                logging.info(
                    f"Using directory {filename} for input, "
                    f"as it contains a model.pnml file.")
                break
            else:
                parts = os.listdir(filename)
                if len(parts) == 1:
                    filename = filename + "/" + parts[0]
                else:
                    logging.error(
                        f"Cannot use directory {filename} for input, "
                        f"as it does not contain a model.pnml file.")
                    return None
        else:
            logging.error(
                f"Cannot use directory {filename} for input, "
                f"as it does not contain a model.pnml file.")
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
    if arguments.exclude is None:
        arguments.exclude = []
    else:
        arguments.exclude = sorted(arguments.exclude.split(","))
    if arguments.forget is None:
        arguments.forget = []
    else:
        arguments.forget = sorted(arguments.forget.split(","))
    # Compute prefix for generated files:
    hasher = hashlib.md5()
    with open(arguments.characteristics, "rb") as hinput:
        hasher.update(hinput.read())
    characteristics_hash = hasher.hexdigest()
    hasher = hashlib.md5()
    with open(arguments.results, "rb") as hinput:
        hasher.update(hinput.read())
    results_hash = hasher.hexdigest()
    as_json = json.dumps({
        "characteristics": characteristics_hash,
        "results": results_hash,
        "duplicates": arguments.duplicates,
        "exclude": arguments.exclude,
        "forget": arguments.forget,
        "training": arguments.training,
        "year": arguments.year,
    }, sort_keys=True)
    hasher = hashlib.md5()
    hasher.update(bytearray(as_json, "utf8"))
    prefix = hasher.hexdigest()[:8]
    logging.info(f"Prefix is {prefix}.")
    with open(f"{arguments.data}/{prefix}-configuration.json", "w") as output:
        output.write(as_json)
    # Load data:
    data = Data({
        "characteristics": arguments.characteristics,
        "results": arguments.results,
        "renaming": RENAMING,
        "exclude": arguments.exclude,
        "year": arguments.year,
    })
    options = {
        "Choice": True,
        "Duplicates": arguments.duplicates,
        "Output Trees": arguments.output_trees,
        "Directory": arguments.data,
        "Prefix": prefix,
        "Forget": arguments.forget,
        "Training": arguments.training,
        "Score": arguments.score,
    }
    # Read data:
    data.characteristics()
    # Compute the characteristics for models:
    characteristics_of(data, options)
    # Use results:
    data.results()
    examinations = {x["Examination"] for x in data.results()}
    tools = {x["Tool"] for x in data.results()}
    # Compute maximum score:
    maxs = max_score(data, options)
    total_score = 0
    for _, subscore in maxs.items():
        total_score += subscore
    logging.info(f"Maximum score is {total_score}:")
    for examination in examinations:
        score = maxs[examination]
        logging.info(f"* {examination}: {score}")
    # Extract known data:
    known_data = known(data)
    with open(f"{arguments.data}/{prefix}-known.json", "w") as output:
        json.dump(known_data, output)
    # Extract learned data:
    learned_data, values = learned(data, options)
    with open(f"{arguments.data}/{prefix}-values.json", "w") as output:
        json.dump(values.items, output)
    # Compute scores for tools:
    for tool in sorted(tools):
        logging.info(f"Computing score of tool: {tool}.")
        score = score_of(data, tool, options)
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
        ratio = math.ceil(100*total/total_score)
        logging.info(f"  Score: {total} / {total_score} ({ratio}%)")
    with open(f"{arguments.data}/{prefix}-learned.json", "w") as output:
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
    for examination in sorted(examinations):
        subscore = maxs[examination]
        logging.info(f"In {examination}, maximum score {subscore}:")
        for element in [x for x in srt if x["Examination"] == examination]:
            score = element["Score"]
            name = element["Name"]
            if score > 0:
                ratio = math.ceil(100*score/subscore)
                logging.info(f"* {score} / {subscore} ({ratio}%) "
                             f"for {name}.")


def do_run(arguments):
    """
    Main function for the run command.
    """
    logging.info(f"Prefix is {arguments.prefix}.")
    # Load known info:
    logging.info(
        f"Reading known information "
        f"in {arguments.data}/{arguments.prefix}-known.json."
    )
    with open(f"{arguments.data}/{arguments.prefix}-known.json", "r") as i:
        known_data = json.load(i)
    # Load learned info:
    logging.info(
        f"Reading learned information "
        f"in {arguments.data}/{arguments.prefix}-learned.json."
    )
    with open(f"{arguments.data}/{arguments.prefix}-learned.json", "r") as i:
        learned_data = json.load(i)
    # Load translations:
    logging.info(
        f"Reading value translations "
        f"in {arguments.data}/{arguments.prefix}-values.json."
    )
    with open(f"{arguments.data}/{arguments.prefix}-values.json", "r") as i:
        translations = json.load(i)
    values = Values(translations)
    # Find input:
    directory = unarchive(arguments.input)
    if directory is None:
        sys.exit(1)
    if arguments.instance is None:
        instance = pathlib.PurePath(directory).stem
    else:
        instance = arguments.instance
    split = re.search(r"([^-]+)\-([^-]+)\-([^-]+)$", instance)
    model = split.group(1)
    logging.info(f"Using {instance} as instance name.")
    logging.info(f"Using {model} as model name.")
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
        if known_data[arguments.examination] is not None:
            if known_data[arguments.examination][instance] is not None:
                known_tools = known_data[arguments.examination][instance]
            elif known_data[arguments.examination][model] is not None:
                known_tools = known_data[arguments.examination][model]
    if known_tools is None:
        logging.warning(
            f"Cannot find known information "
            f"for examination {arguments.examination} "
            f"on instance {instance} or model {model}.")
    # Set algorithm:
    learned_tools = None
    if arguments.algorithm:
        algorithm = sorted(
            [x for x in learned_data
             if x["Algorithm"] == arguments.algorithm],
            key=lambda e: e[arguments.examination],
            reverse=True,
        )[0]["Algorithm"]
        filename = f"{arguments.data}/{arguments.prefix}-learned.{algorithm}.p"
        with open(filename, "rb") as i:
            model = pickle.load(i)
    else:
        algorithm = sorted(
            [x for x in learned_data if x["Is-Algorithm"]],
            key=lambda e: e[arguments.examination],
            reverse=True,
        )[0]["Algorithm"]
        filename = f"{arguments.data}/{arguments.prefix}-learned.{algorithm}.p"
        with open(filename, "rb") as i:
            model = pickle.load(i)
    logging.info(f"Using algorithm or tool {algorithm}.")
    # Find learned tools:
    is_colored = read_boolean(f"{directory}/iscolored")
    if is_colored:
        has_pt = read_boolean(f"{directory}/equiv_pt")
    else:
        has_colored = read_boolean(f"{directory}/equiv_col")
    with open(f"{directory}/GenericPropertiesVerdict.xml", "r") as i:
        verdict = xmltodict.parse(i.read())
    characteristics = {
        "Examination": arguments.examination,
        "Place/Transition": (not is_colored) or has_pt,
        "Colored": is_colored or has_colored,
        "Relative-Time": 1,  # FIXME
        "Relative-Memory": 1,  # FIXME
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
                        f"for examination {arguments.examination} "
                        f"on instance {instance} or model {model}.")
    elif learned_tools is None:
        logging.warning(f"No learned information "
                        f"for examination {arguments.examination} "
                        f"on instance {instance} or model {model}.")
    # Run the tools:
    if arguments.cheat and known_tools is not None:
        tools = known_tools
    elif arguments.cheat and learned_tools is not None:
        tools = learned_tools
    elif learned_tools is not None:
        tools = learned_tools
    else:
        logging.error(f"DO_NOT_COMPETE")
        sys.exit(1)
    path = os.path.abspath(directory)
    # Load docker client:
    client = docker.from_env()
    for entry in tools:
        tool = entry["Tool"]
        logging.info(f"{arguments.examination} {tool} {instance}...")
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
                "BK_EXAMINATION": f"{arguments.examination}",
                "BK_TIME_CONFINEMENT": "3600",
                "BK_INPUT": f"{instance}",
                "BK_TOOL": tool.lower(),
            },
        )
        for line in container.logs(stream=True):
            logging.info(line.decode("UTF-8").strip())
        result = container.wait()
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
        "year": arguments.year,
        "exclude": [],
    })
    # Read data:
    data.characteristics()
    data.results()
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
                for line in container.logs(stream=True):
                    logging.info(line.decode("UTF-8").strip())
                result = container.wait()
                container.remove()
                tested[examination][tool] = result["StatusCode"] == 0
            except docker.errors.NotFound:
                logging.warning(f"Docker image does not exist.")
                tested[examination][tool] = False
    for examination, subresults in tested.items():
        logging.info(f"Tests for {examination}:")
        for tool, value in subresults.items():
            logging.info(f"  {tool}: {value}")


def do_experiment(arguments):
    """
    Main function for the experiment command.
    """
    # Load data:
    data = Data({
        "characteristics": arguments.characteristics,
        "results": arguments.results,
        "renaming": RENAMING,
        "exclude": [],
        "year": arguments.year,
    })
    # Read data:
    data.characteristics()
    # Use results:
    data.results()
    result = []
    if arguments.training:
        for value in range(0, 100, 10):
            training = 1 - (value / 100)
            logging.info(f"Running experiment with {training} training.")
            options = {
                "Duplicates": arguments.duplicates,
                "Training": training,
                "Forget": [],
                "Score": arguments.score,
            }
            subresult = learned(data, options)[0]
            for entry in subresult:
                for okey, ovalue in options.items():
                    entry[okey] = ovalue
                result.append(entry)
    if arguments.forget:
        characteristics = []
        for characteristic in CHARACTERISTICS:
            if characteristic not in REMOVE:
                characteristics.append(characteristic)
        for forget_n in range(0, len(characteristics)+1):
            for _ in range(0, arguments.forget):
                random.shuffle(characteristics)
                forget = characteristics[:forget_n]
                logging.info(f"Running experiment forgetting {forget}.")
                options = {
                    "Duplicates": arguments.duplicates,
                    "Training": 1.0,
                    "Forget": forget,
                    "Score": arguments.score,
                }
                subresult = learned(data, options)[0]
                for entry in subresult:
                    for okey, ovalue in options.items():
                        entry[okey] = ovalue
                    result.append(entry)
    with open(f"{arguments.data}/experiment.json", "w") as output:
        output.write(json.dumps(result, sort_keys=True))


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
    help="path to the results of the model checking contest",
    type=str,
    dest="results",
    default=os.getcwd() + "/results.csv",
)
EXTRACT.add_argument(
    "--characteristics",
    help="path to the model characteristics from the Petri net repository",
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
EXTRACT.add_argument(
    "--exclude",
    help="Exclude tools (comma separated)",
    dest="exclude",
    default=None,
)
EXTRACT.add_argument(
    "--training",
    help="ratio of models to use during training (for instance 0.5)",
    dest="training",
    type=float,
    default=1,
)
EXTRACT.add_argument(
    "--score",
    help="score computation type (mcc or time)",
    dest="score",
    type=str,
    default="mcc",
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
TEST.add_argument(
    "--exclude",
    help="Exclude tools (comma separated)",
    dest="exclude",
    default=None,
)
TEST.set_defaults(func=do_test)


def default_prefix():
    """
    Extracts the default prefix.
    """
    bk_tool = os.getenv("BK_TOOL")
    if bk_tool is not None:
        search = re.search(r"^(mcc|irma)4mcc-(.*)$", bk_tool)
        if search is None:
            result = None
        else:
            result = search.group(2)
    else:
        prefixes = []
        for filename in os.listdir(os.getcwd()):
            if filename.endswith("-configuration.json"):
                search = re.search(r"^([^-]+)-configuration.json$", filename)
                prefixes.append(search.group(1))
        if prefixes:
            result = sorted(prefixes)[0]
        else:
            result = None
    return result


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
    "--prefix",
    help="Prefix of generated files",
    dest="prefix",
    default=default_prefix(),
)
RUN.add_argument(
    "--cheat",
    help="Cheat using known information",
    dest="cheat",
    action="store_true",
)
RUN.set_defaults(func=do_run)

EXPERIMENT = SUBPARSERS.add_parser(
    "experiment",
    description="Experiment",
)
EXPERIMENT.add_argument(
    "--results",
    help="path to the results of the model checking contest",
    type=str,
    dest="results",
    default=os.getcwd() + "/results.csv",
)
EXPERIMENT.add_argument(
    "--characteristics",
    help="path to the model characteristics from the Petri net repository",
    type=str,
    dest="characteristics",
    default=os.getcwd() + "/characteristics.csv",
)
EXPERIMENT.add_argument(
    "--year",
    help="Use results for a specific year (YYYY format).",
    type=int,
    dest="year",
)
EXPERIMENT.add_argument(
    "--duplicates",
    help="Allow duplicate entries",
    dest="duplicates",
    action="store_true"
)
EXPERIMENT.add_argument(
    "--forget",
    help="Iterations in forgetting of characteristics",
    type=int,
    dest="forget",
    default=0,
)
EXPERIMENT.add_argument(
    "--training",
    help="Vary the training rate",
    dest="training",
    action="store_true",
)
EXPERIMENT.add_argument(
    "--score",
    help="score computation type (mcc or time)",
    dest="score",
    type=str,
    default="mcc",
)
EXPERIMENT.set_defaults(func=do_experiment)


ARGUMENTS = PARSER.parse_args()
if "func" in ARGUMENTS:
    ARGUMENTS.func(ARGUMENTS)
else:
    PARSER.print_usage()
