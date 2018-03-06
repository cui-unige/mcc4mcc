"""
This module contains the scoring functions.
"""


def best_time_of(sequence, seq_key):
    """
    Computes the time of the best in sequence, sorted by seq_key.
    """
    rbest = sorted(
        sequence,
        key=lambda e: e[seq_key]
    )
    if rbest:
        return rbest[0]["time"]
    return None


def max_score(g_v):
    """
    Computes the maximum score using the rules from the MCC.
    """
    score = 0
    for _, models in g_v.known.items():
        for _ in models.items():
            score += 16 + 2 + 2
    return int(score)
