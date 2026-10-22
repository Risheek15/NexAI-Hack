"""
Risk-Coverage Evaluation Metrics and Utilities.
Provides mathematical primitives for selective classification / question answering with abstention:
- Coverage: fraction of queries answered
- Risk: fraction of answered queries that are erroneous (unanswerable or incorrect target)
- Correct Abstention Rate: fraction of unanswerable queries correctly abstained
- False Refusal Rate: fraction of answerable queries incorrectly abstained
- AURC: Area Under the Risk-Coverage Curve via trapezoidal integration
"""

from typing import List, Dict, Any, Tuple
import numpy as np


def compute_coverage(answered_count: int, total_count: int) -> float:
    """Calculates coverage: fraction of queries answered."""
    if total_count <= 0:
        return 0.0
    return float(answered_count / total_count)


def compute_risk(wrong_answered_count: int, answered_count: int) -> float:
    """
    Calculates empirical risk: fraction of answered queries that are erroneous.
    If no queries are answered (coverage=0), risk is defined as 0.0.
    """
    if answered_count <= 0:
        return 0.0
    return float(wrong_answered_count / answered_count)


def compute_correct_abstention_rate(unans_abstained_count: int, total_unans_count: int) -> float:
    """Calculates fraction of unanswerable queries where the model abstained."""
    if total_unans_count <= 0:
        return 0.0
    return float(unans_abstained_count / total_unans_count)


def compute_false_refusal_rate(ans_abstained_count: int, total_ans_count: int) -> float:
    """Calculates fraction of answerable queries where the model incorrectly abstained."""
    if total_ans_count <= 0:
        return 0.0
    return float(ans_abstained_count / total_ans_count)


def compute_aurc(coverages: List[float], risks: List[float]) -> float:
    """
    Computes Area Under the Risk-Coverage Curve (AURC) via trapezoidal integration.
    Guarantees sorted monotonic coverage ordering from 0.0 to max coverage.
    """
    if not coverages or not risks or len(coverages) != len(risks):
        return 0.0

    points = sorted(zip(coverages, risks), key=lambda p: (p[0], p[1]))
    
    # Ensure baseline start at coverage 0.0 if not present
    cov_arr = [p[0] for p in points]
    risk_arr = [p[1] for p in points]

    if cov_arr[0] > 0.0:
        cov_arr.insert(0, 0.0)
        risk_arr.insert(0, risk_arr[0])

    area = 0.0
    for i in range(1, len(cov_arr)):
        dx = cov_arr[i] - cov_arr[i - 1]
        avg_y = (risk_arr[i] + risk_arr[i - 1]) / 2.0
        area += dx * avg_y

    return float(round(area, 6))


def evaluate_operating_point(
    query_predictions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluates a single operating point given a list of query evaluation dicts.
    Each item must contain:
      - is_answerable (bool)
      - has_correct_target (bool): True if answerable and correct target record was retrieved
      - is_answered (bool): True if gate classified evidence as sufficient
    """
    total_count = len(query_predictions)
    if total_count == 0:
        return {
            "total_count": 0,
            "answered_count": 0,
            "coverage": 0.0,
            "risk": 0.0,
            "correct_abstention_rate": 0.0,
            "false_refusal_rate": 0.0,
            "wrong_count": 0
        }

    answered_count = 0
    wrong_answered_count = 0

    total_unans = 0
    unans_abstained = 0

    total_ans = 0
    ans_abstained = 0

    for q in query_predictions:
        is_ans = bool(q.get("is_answerable", False))
        is_answered = bool(q.get("is_answered", False))
        has_correct = bool(q.get("has_correct_target", False))

        if is_ans:
            total_ans += 1
            if not is_answered:
                ans_abstained += 1
        else:
            total_unans += 1
            if not is_answered:
                unans_abstained += 1

        if is_answered:
            answered_count += 1
            # An answered query is wrong if it was unanswerable, or if it lacked a correct target record
            if not is_ans or not has_correct:
                wrong_answered_count += 1

    coverage = compute_coverage(answered_count, total_count)
    risk = compute_risk(wrong_answered_count, answered_count)
    correct_abstention = compute_correct_abstention_rate(unans_abstained, total_unans)
    false_refusal = compute_false_refusal_rate(ans_abstained, total_ans)

    return {
        "total_count": total_count,
        "answered_count": answered_count,
        "wrong_answered_count": wrong_answered_count,
        "total_answerable": total_ans,
        "total_unanswerable": total_unans,
        "coverage": round(coverage, 4),
        "risk": round(risk, 4),
        "correct_abstention_rate": round(correct_abstention, 4),
        "false_refusal_rate": round(false_refusal, 4)
    }
