#! /usr/bin/env python3

"""
Extract data from the Model Checking Contest results,
generate exact choice algorithms,
and learn from data for approximate algorithm.
"""

import argparse
import logging
import os
from ml_algorithm import init_algorithms
from global_variables import GlobalVariales
from processing import read_characteristics, read_results, set_techniques
from processing import rename_tools, sort_data, analyze_known, translate
from score import compute_scores, analyze_learned, analyze_useless


if __name__ == "__main__":

    # It parses the cli arguments
    PARSER = argparse.ArgumentParser(
        description="Data extractor for the model checker collection"
    )
    PARSER.add_argument(
        "--results",
        help="results of the model checking contest",
        type=str,
        dest="results",
        default=os.getcwd() + "/results.csv",
    )
    PARSER.add_argument(
        "--characteristics",
        help="model characteristics from the Petri net repository",
        type=str,
        dest="characteristics",
        default=os.getcwd() + "/characteristics.csv",
    )
    PARSER.add_argument(
        "--known",
        help="data known from models",
        type=str,
        dest="known",
        default=os.getcwd() + "/known.json",
    )
    PARSER.add_argument(
        "--learned",
        help="data learned from models",
        type=str,
        dest="learned",
        default=os.getcwd() + "/learned.json",
    )
    PARSER.add_argument(
        "--iterations",
        help="Number of iterations",
        type=int,
        dest="iterations",
        default=0,
    )
    PARSER.add_argument(
        "--distance",
        help="Allowed distance from the best tool (in percent)",
        type=float,
        dest="distance",
    )
    PARSER.add_argument(
        "--duplicates",
        help="Allow duplicate entries",
        type=bool,
        dest="duplicates",
        default=False,
    )
    PARSER.add_argument(
        "--score",
        help="Compute score in the Model Checking Contest",
        type=bool,
        dest="mcc_score",
        default=True,
    )
    PARSER.add_argument(
        "--useless",
        help="Compute useless characteristics",
        type=bool,
        dest="useless",
        default=False,
    )
    PARSER.add_argument(
        "--output-dt",
        help="Output the graph of trained decision tree.",
        type=bool,
        dest="output_dt",
        default=False,
    )
    ARGUMENTS = PARSER.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    translate.ITEMS = {}
    translate.NEXT_ID = 10

    # Init the class containing all the global variables.
    GV = GlobalVariales()
    GV.algorithms = init_algorithms(ARGUMENTS)

    logging.info(
        f"Reading model characteristics from '{ARGUMENTS.characteristics}'.")
    read_characteristics(ARGUMENTS, GV)

    logging.info(f"Reading mcc results from '{ARGUMENTS.results}'.")
    read_results(ARGUMENTS, GV)

    logging.info(f"Setting all techniques to Boolean values.")
    set_techniques(GV)

    logging.info(f"Renaming tools.")
    rename_tools(GV)

    GV.size = len(GV.results)

    logging.info(f"Sorting data.")
    GV.size = sort_data(GV)
    GV.distance = ARGUMENTS.distance

    logging.info(f"Analyzing known data.")
    analyze_known(GV)

    if ARGUMENTS.mcc_score:
        logging.info(f"Computing scores.")
        compute_scores(GV)

    translate.ITEMS = {
        False: -1,
        None: 0,
        True: 1,
    }

    logging.info(f"Analyzing learned data.")
    analyze_learned(ARGUMENTS, GV)

    if ARGUMENTS.useless and ARGUMENTS.mcc_score:
        logging.info(f"Analyzing useless characteristics.")
        analyze_useless(ARGUMENTS, GV)
