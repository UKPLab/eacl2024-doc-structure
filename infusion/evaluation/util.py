from typing import Dict


def precision(confusion: Dict[str, int]) -> float:
    if confusion["TP"] == 0 and confusion["FP"] == 0:
        return 0  # TODO: is this correct?
    else:
        return confusion["TP"] / (confusion["TP"] + confusion["FP"])


def recall(confusion: Dict[str, int]) -> float:
    if confusion["TP"] == 0 and confusion["FN"] == 0:
        return 0  # TODO: is this correct?
    else:
        return confusion["TP"] / (confusion["TP"] + confusion["FN"])


def f1_score(confusion: Dict[str, int]) -> float:
    prec = precision(confusion)
    reca = recall(confusion)
    if prec + reca == 0:
        return 0
    else:
        return 2 * prec * reca / (prec + reca)
