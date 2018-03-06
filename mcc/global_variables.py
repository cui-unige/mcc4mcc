
class GlobalVariales():
    """docstring for GlobalVariales"""

    def __init__(self, ALGORITHMS={}):
        self.CHARACTERISTIC_KEYS = [
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
        self.RESULT_KEYS = [
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
        self.TO_DROP = [
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
        self.TOOLS_RENAME = {
            "tapaalPAR": "tapaal",
            "tapaalSEQ": "tapaal",
            "tapaalEXP": "tapaal",
            "sift": "tina",
            "tedd": "tina",
        }
        self.ALGORITHMS = ALGORITHMS
        self.TECHNIQUES = {}
        self.CHARACTERISTICS = {}
        self.TOOLS = {}
