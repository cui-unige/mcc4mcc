#! /usr/bin/env python3

if __name__ == "__main__":

    import argparse
    import io
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
        "--known",
        help    = "data known from models",
        type    = str,
        dest    = "known",
        default = os.getcwd () + "/known.json",
    )
    parser.add_argument (
        "--learned",
        help    = "data learned from models",
        type    = str,
        dest    = "learned",
        default = os.getcwd () + "/learned.json",
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
    parser.add_argument (
        "-v", "--verbose",
        help   = "input directory or archive",
        action = "store_true",
        dest   = "verbose",
    )
    arguments = parser.parse_args ()
    if arguments.verbose:
        logging.basicConfig (level = logging.INFO)
    else:
        logging.basicConfig (level = logging.WARNING)

    logging.info (f"Reading known information in '{arguments.known}'.")
    known = json.load (io.FileIO (arguments.known))

    logging.info (f"Reading learned information in '{arguments.learned}'.")
    learned = json.load (io.FileIO (arguments.learned))

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
        if x in learned.translation:
            return learned.translation [x]
        return x

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
        logging.warning (f"Cannot find known information for examination {examination} on {model} or {instance}.")
        # TODO: read model, extract characteristics, translate them, and find the tool to use.
        model = pickle.load (open (f"learned.{algorithm}.p", "rb"))
        model.predict () # http://scikit-learn.org/stable/modules/model_persistence.html
        sys.exit (1)

    log = os.getenv ("BK_LOG_FILE")
    if log is None:
        log = tempfile.TemporaryFile ()

    if not tools:
        logging.error (f"DO NOT COMPETE")
        sys.exit (1)

    success = None
    path    = os.path.abspath (arguments.input)
    for x in tools:
        tool = x ["tool"]
        logging.info (f"Starting tool '{tool}'...")
        success = subprocess.call ([
            "docker", "run",
            "--volume", f"{path}:/mcc-data",
            "--workdir", "/mcc-data",
            "--env", f"BK_TOOL={tool}",
            "--env", f"BK_EXAMINATION={examination}",
            "--env", f"BK_INPUT={last}",
            "--env", f"BK_LOG_FILE={log}",
            f"mcc/{tool}",
        ])
        if success == 0:
            break
    if success != 0:
        logging.error (f"CANNOT COMPUTE")
        sys.exit (1)
