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
    try:
        return int (x)
    except ValueError:
        pass
    try:
        return float (x)
    except ValueError:
        pass
    return x

if __name__ == "__main__":

    import argparse
    import csv
    import io
    import json
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

    logging.info (f"Sorting data.")
    size      = len (results)
    data      = {}
    tool_year = {}
    with tqdm (total = len (results)) as counter:
        for _, entry in results.items ():
            if entry ["Examination"] not in data:
                data [entry ["Examination"]] = {}
            examination = data [entry ["Examination"]]
            if entry ["Model Id"] not in examination:
                examination [entry ["Model Id"]] = {}
            model = examination [entry ["Model Id"]]
            if entry ["Instance"] not in model:
                model [entry ["Instance"]] = {}
            instance = model [entry ["Instance"]]
            if entry ["Tool"] not in instance:
                instance [entry ["Tool"]] = {}
            tool = instance [entry ["Tool"]]
            if entry ["Tool"] not in tool_year:
                tool_year [entry ["Tool"]] = 0
            if entry ["Year"] > tool_year [entry ["Tool"]]:
                tool_year [entry ["Tool"]] = entry ["Year"]
            if entry ["Year"] in tool:
                size -= 1
                if entry ["Clock Time"] < tool [entry ["Year"]] ["Clock Time"]:
                    tool [entry ["Year"]] = entry
            else:
                tool [entry ["Year"]] = entry
            counter.update (1)

    logging.info (f"Analyzing known data.")
    known = {}
    with tqdm (total = size) as counter:
        for examination, models in data.items ():
            known [examination] = {}
            known_e = known [examination]
            for model, instances in models.items ():
                known_e [model] = {}
                known_m = known_e [model]
                subresults = {}
                for instance, tools in instances.items ():
                    known_m [instance] = {}
                    known_i = known_m [instance]
                    subsubresults = {}
                    for tool, years in tools.items ():
                        if tool not in subresults:
                            subresults [tool] = {
                                "count" : 0,
                                "time"  : 0,
                                "memory": 0,
                            }
                        for year, entry in years.items ():
                            if year == tool_year [tool]:
                                subsubresults [tool] = {
                                    "time"  : entry ["Clock Time"],
                                    "memory": entry ["Memory"],
                                }
                                subresults [tool] ["count"]  += 1
                                subresults [tool] ["time"]   += entry ["Clock Time"]
                                subresults [tool] ["memory"] += entry ["Memory"]
                            counter.update (1)
                    s = sorted (subsubresults.items (), key = lambda e: (e [1] ["time"], e [1] ["memory"]))
                    known_i ["sorted"] = [ { "tool": x [0], "time": x [1] ["time"], "memory": x [1] ["memory"] } for x in s]
                s = sorted (subresults.items (), key = lambda e: (- e [1] ["count"], e [1] ["time"], e [1] ["memory"]))
                known_m ["sorted"] = [ { "tool": x [0], "count": x [1] ["count"], "time": x [1] ["time"], "memory": x [1] ["memory"] } for x in s]
    with open ("known.json", "w") as output:
        json.dump (known, output)
