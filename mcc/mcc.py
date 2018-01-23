#! /usr/bin/env python3

if __name__ == "__main__":

    import argparse
    import io
    import json
    import logging
    import os
    import pathlib
    import re
    import subprocess
    import sys
    import tempfile
    import tarfile

    parser = argparse.ArgumentParser (description="Model Checker Collection for the Model Checking Contest")
    parser.add_argument (
        "-i", "--input",
        help    = "input directory or archive",
        type    = str,
        dest    = "input",
        default = os.getcwd (),
    )
    parser.add_argument (
        "-k", "--known",
        help    = "data known from models",
        type    = str,
        dest    = "known",
        default = os.getcwd () + "/known.json",
    )
    parser.add_argument (
        "-l", "--learned",
        help    = "data learned from models",
        type    = str,
        dest    = "learned",
        default = os.getcwd () + "/learned.json",
    )
    parser.add_argument (
        "-e", "--examination",
        help    = "examniation type",
        type    = str,
        dest    = "examination",
        default = os.getenv ("BK_EXAMINATION"),
    )
    parser.add_argument (
        "-t", "--tool",
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
    if split == None:
        instance = last
        model    = last
    else:
        instance = last
        model    = split.group (1)
    logging.info (f"Using '{instance}' as instance name.")
    logging.info (f"Using '{model}' as model name.")


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
        logging.warning (f"Cannot find known information for {examination} on {model} or {instance}.")
        sys.exit (1)

    log = os.getenv ("BK_LOG_FILE")
    if log == None:
        log = tempfile.TemporaryFile ()

    if len (tools) == 0:
        logging.error (f"DO NOT COMPETE")
        sys.exit (1)

    success = None
    path    = os.path.abspath (arguments.input)
    for tool in tools:
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
