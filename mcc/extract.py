#! /usr/bin/env python3

CHARACTERISTICS = [
    "Id",
    "Type",
    "Fixed size",
    "Parameterised",
    "Connected",
    "Conservative",
    "Deadlock",
    "Extended Free Choice",
    "Live",
    "Loop Free",
    "Marked Graph",
    "Nested Units",
    "Ordinary",
    "Quasi Live",
    "Reversible",
    "Safe",
    "Simple Free Choice",
    "Sink Place",
    "Sink Transition",
    "Source Place",
    "Source Transition",
    "State Machine",
    "Strongly Connected",
    "Sub-Conservative",
    "Origin",
    "Submitter",
    "Year",
]
RESULTS = [
    "Year",
    "Tool",
    "Instance",
    "Examination",
    "Cores",
    "Time OK",
    "Memory OK",
    "Results",
    "Techniques",
    "Memory",
    "CPU Time",
    "Clock Time",
    "IO Time",
    "Status",
    "Id",
]

def value_of (x):
    if x in [ "True", "Yes", "OK" ]:
        return True
    elif x in [ "False", "None" ]:
        return False
    elif x == "Unknown":
        return None
    else:
        try:
            return int (x)
        except ValueError:
            return x

if __name__ == "__main__":

    import argparse
    import csv
    import logging
    import os
    import re
    from tqdm import tqdm

    parser = argparse.ArgumentParser (
        description = "..." # FIXME
    )
    parser.add_argument (
        "--results",
        help    = "results of the model checking contest",
        type    = str,
        dest    = "results",
        default = os.getcwd () + "/results.csv",
    )
    parser.add_argument (
        "--characteristics",
        help    = "model characteristics from the Petri net repository",
        type    = str,
        dest    = "characteristics",
        default = os.getcwd () + "/characteristics.csv",
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

    techniques = {}

    characteristics = {}
    logging.info (f"Reading model characteristics from '{arguments.characteristics}'.")
    with tqdm (total = sum (1 for line in open (arguments.characteristics))) as counter:
        with open (arguments.characteristics) as data:
            data.readline () # skip the title line
            reader = csv.reader (data)
            for row in reader:
                entry = {}
                for i, characteristic in enumerate (CHARACTERISTICS):
                    entry [characteristic] = value_of (row [i])
                entry ["Place/Transition"] = True if re.search ("PT"     , entry ["Type"]) else False
                entry ["Colored"         ] = True if re.search ("COLORED", entry ["Type"]) else False
                del entry ["Type"      ]
                del entry ["Fixed size"]
                del entry ["Origin"    ]
                del entry ["Submitter" ]
                del entry ["Year"      ]
                characteristics [entry ["Id"]] = entry
                counter.update (1)

    results = {}
    logging.info (f"Reading mcc results from '{arguments.results}'.")
    with tqdm (total = sum (1 for line in open (arguments.results))) as counter:
        with open (arguments.results) as data:
            data.readline () # skip the title line
            reader = csv.reader (data)
            for row in reader:
                entry = {}
                for i, result in enumerate (RESULTS):
                    entry [result] = value_of (row [i])
                if  entry ["Time OK"  ] \
                and entry ["Memory OK"] \
                and entry ["Status"   ] == "normal" \
                and entry ["Results"  ] != "DNC" \
                and entry ["Results"  ] != "DNF" \
                and entry ["Results"  ] != "CC":
                    results [entry ["Id"]] = entry
                    for technique in re.findall (r"([A-Z_]+)", entry ["Techniques"]):
                        techniques [technique] = True
                        entry [technique] = True
                    entry ["Surprise"] = True if re.search (r"^S_", entry ["Instance"]) else False
                    if entry ["Surprise"]:
                        entry ["Instance"] = re.search (r"^S_(.*)$", entry ["Instance"]).group (1)
                    split = re.search (r"([^-]+)\-([^-]+)\-([^-]+)$", entry ["Instance"])
                    if split is None:
                        entry ["Model Id"] = entry ["Instance"]
                    else:
                        entry ["Model Id"] = split.group (1)
                    if entry ["Model Id"] in characteristics:
                        model = characteristics [entry ["Model Id"]]
                        for key in model.keys ():
                            if key != "Id":
                                entry [key] = model [key]
                    del entry ["Time OK"   ]
                    del entry ["Memory OK" ]
                    del entry ["CPU Time"  ]
                    del entry ["Cores"     ]
                    del entry ["IO Time"   ]
                    del entry ["Results"   ]
                    del entry ["Status"    ]
                    del entry ["Techniques"]
                counter.update (1)

    logging.info (f"Setting all techniques to Boolean values.")
    with tqdm (total = len (results)) as counter:
        for key, entry in results.items ():
            for technique in techniques:
                if technique not in entry:
                    entry [technique] = False
            counter.update (1)
