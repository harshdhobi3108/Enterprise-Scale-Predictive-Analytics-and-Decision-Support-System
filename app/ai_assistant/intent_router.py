from typing import Literal

IntentType = Literal[
    "greeting",
    "revenue_analysis",
    "churn_analysis",
    "delivery_analysis",
    "segmentation_analysis",
    "general"
]

def detect_intent(query: str) -> IntentType:
    q = query.lower()

    if any(word in q for word in ["hi", "hello", "hey"]):
        return "greeting"

    if "revenue" in q:
        return "revenue_analysis"

    if "churn" in q or "retention" in q:
        return "churn_analysis"

    if "delivery" in q or "delay" in q:
        return "delivery_analysis"

    if "segment" in q or "customer group" in q:
        return "segmentation_analysis"

    return "general"