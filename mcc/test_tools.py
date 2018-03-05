#! /usr/bin/env python3

"""
Test the integration of tools within mcc4mcc.
"""

if __name__ == "__main__":

    import argparse
    import json
    import getpass
    import logging
    import os
    import readline
    import subprocess
    import tempfile
    import tarfile
    import docker

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
    ARGUMENTS = PARSER.parse_args()
    logging.basicConfig(level=logging.INFO)

    logging.info(
        f"Reading known information in '{ARGUMENTS.data}/known.json'.")
    with open(f"{ARGUMENTS.data}/known.json", "r") as i:
        KNOWN = json.load(i)

    client = docker.from_env()
    client.login(
        username=input("Docker username: "),
        password=getpass.getpass("Docker password: "),
    )

    PATH = os.path.abspath(ARGUMENTS.input)
    TESTED = {}
    for examination, models in KNOWN.items():
        if examination not in TESTED:
            TESTED[examination] = {}
        for model, instances in models.items():
            for instance, entries in instances.items():
                if instance == "sorted":
                    continue
                for entry in entries["sorted"]:
                    tool = entry["tool"]
                    if tool in TESTED[examination]:
                        continue
                    directory = f"{PATH}/{instance}"
                    if not os.path.isdir(directory):
                        tarname = f"{PATH}/{instance}.tgz"
                        logging.info(
                            f"Extracting archive {tarname} "
                            f"to temporary directory {directory}.")
                        with tarfile.open(name=tarname) as tar:
                            tar.extractall(path=PATH)
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
                                "BK_EXAMINATION": f"{examination}",
                                "BK_TIME_CONFINEMENT": "3600",
                                "BK_INPUT": f"{instance}",
                                "BK_TOOL": tool.lower(),
                            },
                        )
                        logging.info(logs)
                        TESTED[examination][tool] = True
                    except docker.errors.ContainerError as e:
                        logging.error(f"  Failure", e)
                        TESTED[examination][tool] = False
                    except docker.errors.ImageNotFound as e:
                        TESTED[examination][tool] = False
                    except docker.errors.APIError as e:
                        logging.error(f"  Unexpected error", e)
                        TESTED[examination][tool] = False
