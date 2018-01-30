#! /usr/bin/env python3

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

    parser = argparse.ArgumentParser (
        description = "Model Checker Collection for the Model Checking Contest"
    )
    parser.add_argument (
        "--input",
        help    = "input directory or archive",
        type    = str,
        dest    = "input",
        default = os.getcwd (),
    )
    parser.add_argument (
        "--data",
        help    = "directory containing data known and learned from models",
        type    = str,
        dest    = "data",
        default = os.getcwd (),
    )
    parser.add_argument (
        "--algorithm",
        help    = "machine learning algorithm to use",
        type    = str,
        dest    = "algorithm",
    )
    parser.add_argument (
        "--examination",
        help    = "examination type",
        type    = str,
        dest    = "examination",
        default = os.getenv ("BK_EXAMINATION"),
    )
    parser.add_argument (
        "--tool",
        help    = "tool",
        type    = str,
        dest    = "tool",
    )
    arguments = parser.parse_args ()
    logging.basicConfig (level = logging.INFO)

    verdicts = {
        "ORDINARY"            : "Ordinary",
        "SIMPLE_FREE_CHOICE"  : "Simple Free Choice",
        "EXTENDED_FREE_CHOICE": "Extended Free Choice",
        "STATE_MACHINE"       : "State Machine",
        "MARKED_GRAPH"        : "Marked Graph",
        "CONNECTED"           : "Connected",
        "STRONGLY_CONNECTED"  : "Strongly Connected",
        "SOURCE_PLACE"        : "Source Place",
        "SINK_PLACE"          : "Sink Place",
        "SOURCE_TRANSITION"   : "Source Transition",
        "SINK_TRANSITION"     : "Sink Transition",
        "LOOP_FREE"           : "Loop Free",
        "CONSERVATIVE"        : "Conservative",
        "SUBCONSERVATIVE"     : "Sub-Conservative",
        "NESTED_UNITS"        : "Nested Units",
        "SAFE"                : "Safe",
        "DEADLOCK"            : "Deadlock",
        "REVERSIBLE"          : "Reversible",
        "QUASI_LIVE"          : "Quasi Live",
        "LIVE"                : "Live",
    }


    logging.info (f"Reading known information in '{arguments.data}/known.json'.")
    with open (f"{arguments.data}/known.json", "r") as i:
        known = json.load (i)

    logging.info (f"Reading learned information in '{arguments.data}/learned.json'.")
    with open (f"{arguments.data}/learned.json", "r") as i:
        learned = json.load (i)

    if arguments.algorithm is None:
        algorithm = sorted (learned ["algorithms"], key = lambda e: e ["mean"], reverse = True) [0] ["algorithm"]
    else:
        algorithm = arguments.algorithm
    logging.info (f"Using machine learning algorithm '{algorithm}'.")

    while True:
        if os.path.isfile (arguments.input):
            directory = tempfile.TemporaryDirectory ()
            logging.info (f"Extracting archive '{arguments.input}' to temporary directory '{directory}'.")
            with tarfile.open (name = arguments.input) as tar:
                tar.extractall (path = directory)
            arguments.input = directory
        elif os.path.isdir (arguments.input):
            if os.path.isfile (arguments.input + "/model.pnml"):
                logging.info (f"Using directory '{arguments.input}' for input, as it contains a 'model.pnml' file.")
                break
            else:
                logging.error (f"Cannot use directory '{arguments.input}' for input, as it does not contain a 'model.pnml' file.")
                sys.exit (1)
        else:
            logging.error (f"Cannot use directory '{arguments.input}' for input, as it does not contain a 'model.pnml' file.")
            sys.exit (1)

    last  = pathlib.PurePath (arguments.input).stem
    split = re.search (r"([^-]+)\-([^-]+)\-([^-]+)$", last)
    if split is None:
        instance = last
        model    = last
    else:
        instance = last
        model    = split.group (1)
    logging.info (f"Using '{instance}' as instance name.")
    logging.info (f"Using '{model}' as model name.")

    def translate (x):
        if x is None:
            return 0
        if x in learned ["translation"]:
            return learned ["translation"] [x]
        return x

    def translate_back (x):
        for key, value in learned ["translation"].items ():
            if value == x:
                return key
        return None

    def read_boolean (x):
        with open (x, "r") as i:
            x = i.readline ().strip ()
            if x == "TRUE":
                return True
            if x == "FALSE":
                return False
            return None

    examination = arguments.examination
    if arguments.tool != None:
        logging.info (f"Using only the tool '{arguments.tool}'.")
        tools = [ arguments.tool ]
    elif examination in known \
     and model in known [examination] \
     and instance in known [examination] [model]:
        tools = known [examination] [model] [instance] ["sorted"]
    elif examination in known \
     and model in known [examination]:
        tools = known [examination] [model] ["sorted"]
    else:
        logging.warning (f"Cannot find known information for examination '{examination}' on instance '{instance}' or model '{model}'.")
        is_colored = read_boolean (f"{arguments.input}/iscolored")
        if is_colored:
            has_pt = read_boolean (f"{arguments.input}/equiv_pt")
        else:
            has_colored = read_boolean (f"{arguments.input}/equiv_col")
        with open (f"{arguments.input}/GenericPropertiesVerdict.xml", "r") as i:
            verdict = xmltodict.parse (i.read ())
        characteristics = {
            "Examination"     : examination,
            "Place/Transition": (not is_colored) or has_pt,
            "Colored"         : is_colored or has_colored,
        }
        for v in verdict ["toolspecific"] ["verdict"]:
            if v ["@value"] == "true":
                characteristics [verdicts [v ["@reference"]]] = True
            elif v ["@value"] == "false":
                characteristics [verdicts [v ["@reference"]]] = False
            else:
                characteristics [verdicts [v ["@reference"]]] = None
        logging.info (f"Model characteristics are: {characteristics}.")
        with open (f"{arguments.data}/learned.{algorithm}.p", "rb") as i:
            model = pickle.load (i)
        test = {}
        for key, value in characteristics.items ():
            test [key] = translate (value)
        # http://scikit-learn.org/stable/modules/model_persistence.html
        predicted = model.predict (pandas.DataFrame ([test]))
        # FIXME: i am not sure the result is correct, because there is no check
        # that the fields of the characteristic have the same name as the
        # fields that were used during learning.
        tools = [ { "tool": translate_back (predicted [0]) } ]

    # log = os.getenv ("BK_LOG_FILE")
    # if log is None:
    #     log = tempfile.TemporaryFile ()

    if not tools:
        logging.error (f"DO NOT COMPETE")
        sys.exit (1)

    success = None
    path    = os.path.abspath (arguments.input)
    for x in tools:
        tool    = x ["tool"]
        logging.info (f"Starting tool '{tool}'...")
        command = [
            "docker",
            "run",
            "--volume", f"{path}:/mcc-data",
            "--workdir", "/mcc-data",
        ]
        for key, value in os.environ.items ():
            if key.startswith ("BK_"):
                command.append ("--env")
                command.append (f"{key}={value}")
        command.append (f"mcc/{tool}")
        logging.info (f"Running {command}.")
        success = subprocess.call (command)
        if success == 0:
            break
    if success != 0:
        logging.error (f"CANNOT COMPUTE")
        sys.exit (1)
