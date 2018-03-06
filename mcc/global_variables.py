"""
The main code was too coupled with global variables. So this class centrelize
them.
"""


class GlobalVariales():
    """docstring for GlobalVariales"""

    def __init__(self):
        self.characteristic_keys = [
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
        self.result_keys = [
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
        self.to_drop = [
            "Ordinary",
            "Simple Free Choice",
            "Extended Free Choice",
            "State Machine",
            "Marked Graph",
            "Connected",
            "Strongly Connected",
            "Source Place",
            "Sink Place",
            "Source Transition",
            "Sink Transition",
            "Loop Free",
            "Conservative",
            "Sub-Conservative",
            "Nested Units",
            "Safe",
            "Deadlock",
            "Reversible",
            "Quasi Live",
            "Live",
        ]
        self.tools_rename = {
            "tapaalPAR": "tapaal",
            "tapaalSEQ": "tapaal",
            "tapaalEXP": "tapaal",
            "sift": "tina",
            "tedd": "tina",
        }
        self.algorithms = {}
        self.techniques = {}
        self.characteristics = {}
        self.tools = {}
        self.results = {}
        self.data = {}
        self.tool_year = {}
        self.known = {}
        self.max_score = 16 + 2 + 2
        self.scores = {}
        self.learned = []
        self.algorithms_results = []
        self.remove = [
            "Id", "Model Id", "Instance", "Year",
            "Memory", "Clock Time",
            "Parameterised", "Selected", "Surprise"
        ]
