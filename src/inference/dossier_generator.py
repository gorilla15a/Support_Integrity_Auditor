from pathlib import Path

import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "best_deberta"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()


def build_text(
    category,
    channel,
    priority,
    resolution_hours,
    subject,
    description
):

    return f"""
Ticket Category: {category}
Ticket Channel: {channel}
Assigned Priority: {priority}
Resolution Time Hours: {resolution_hours}

Subject:
{subject}

Description:
{description}
"""


def predict_mismatch(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probs = F.softmax(
            outputs.logits,
            dim=1
        )

    prediction = torch.argmax(
        probs,
        dim=1
    ).item()

    confidence = probs.max().item()

    return prediction, confidence


def _generate_dossier_from_row(row, confidence):

    evidence = []

    evidence.append({
        "signal": "llm_score",
        "value": int(row["llm_score"]),
        "weight": "high"
    })

    evidence.append({
        "signal": "cluster_score",
        "value": int(row["cluster_score"]),
        "weight": "medium"
    })

    evidence.append({
        "signal": "resolution_time",
        "value": int(row["Resolution_Time_Hours"]),
        "interpretation":
        (
            "Long resolution duration"
            if row["Resolution_Time_Hours"] > 72
            else "Normal resolution duration"
        )
    })

    evidence.append({
        "signal": "rule_score",
        "value": int(row["rule_score"]),
        "weight": "medium"
    })

    if (
        isinstance(
            row["rule_evidence"],
            str
        )
        and
        row["rule_evidence"].strip()
    ):

        evidence.append({
            "signal": "keyword",
            "value": row["rule_evidence"],
            "weight": "high"
        })

    explanation = (
        f"The ticket was assigned "
        f"{row['Priority_Level']} priority "
        f"but inferred as "
        f"{row['inferred_severity']} severity. "
        f"Multiple independent signals contributed "
        f"to the severity estimate, including "
        f"LLM severity scoring, semantic clustering, "
        f"resolution-time analysis and rule-based "
        f"features. "
        f"The resulting severity delta was "
        f"{row['severity_delta']} and the case "
        f"was categorized as "
        f"{row['mismatch_type']}."
    )

    return {
        "ticket_id": row["Ticket_ID"],
        "assigned_priority": row["Priority_Level"],
        "inferred_severity": row["inferred_severity"],
        "mismatch_type": row["mismatch_type"],
        "severity_delta": int(row["severity_delta"]),
        "feature_evidence": evidence,
        "constraint_analysis": explanation,
        "confidence": round(confidence, 3)
    }


def generate_dossier(*args, **kwargs):
    if len(args) == 2 and isinstance(args[0], dict):
        return _generate_dossier_from_row(args[0], args[1])

    if len(args) == 8:
        category, channel, priority, resolution_hours, subject, description, prediction, confidence = args

        text = f"{subject} {description}".lower()

        severity_map = {
            "Low": 0,
            "Medium": 1,
            "High": 2,
            "Critical": 3
        }

        evidence = []
        keywords = []

        watch_words = [
            "production outage",
            "service outage",
            "server error",
            "500",
            "payment failed",
            "unauthorized access",
            "fraud detected",
            "cannot access",
            "login failed",
            "login error",
            "timeout",
            "api failure",
            "api unavailable"
        ]

        for word in watch_words:
            if word in text:
                keywords.append(word)

        if keywords:
            evidence.append({
                "signal": "keyword",
                "value": ", ".join(keywords[:5]),
                "weight": "high"
            })

        evidence.append({
            "signal": "category",
            "value": category
        })

        evidence.append({
            "signal": "channel",
            "value": channel
        })

        evidence.append({
            "signal": "resolution_time",
            "value": str(resolution_hours),
            "interpretation": (
                "Long resolution duration"
                if resolution_hours > 72
                else "Normal resolution duration"
            )
        })

        inferred_severity = priority
        if prediction == 1:
            if priority == "Critical":
                inferred_severity = "Medium"

            elif priority == "High":
                inferred_severity = "Medium"

            elif priority in ("Low", "Medium"):
                inferred_severity = "High"

        delta = severity_map[inferred_severity] - severity_map[priority]

        if delta > 0:
            mismatch_type = "Hidden Crisis"
        elif delta < 0:
            mismatch_type = "False Alarm"
        else:
            mismatch_type = "Consistent"

        keyword_string = ", ".join(keywords[:3]) if keywords else None

        if mismatch_type == "Hidden Crisis":

            if any(word in text for word in ["fraud detected", "unauthorized access", "breach"]):

                analysis = (
                    f"The ticket was assigned {priority} priority but "
                    f"was inferred as {inferred_severity} severity. "
                    f'The description contains security-related indicators '
                    f'("{keyword_string}"). '
                    f"These observations contributed to a Hidden Crisis "
                    f"classification."
                )

            elif any(word in text for word in ["production outage", "service outage", "500", "server error", "api failure", "api unavailable"]):

                analysis = (
                    f"The ticket was assigned {priority} priority but "
                    f"was inferred as {inferred_severity} severity. "
                    f'The description contains technical failure indicators '
                    f'("{keyword_string}"). '
                    f"These observations contributed to a Hidden Crisis "
                    f"classification."
                )

            elif any(word in text for word in ["login failed", "login error", "cannot access", "unauthorized access"]):

                analysis = (
                    f"The ticket was assigned {priority} priority but "
                    f"was inferred as {inferred_severity} severity. "
                    f'The description contains access-related indicators '
                    f'("{keyword_string}"). '
                    f"These observations contributed to a Hidden Crisis "
                    f"classification."
                )

            elif resolution_hours > 72:

                analysis = (
                    f"The ticket was assigned {priority} priority but "
                    f"was inferred as {inferred_severity} severity. "
                    f"The issue required {resolution_hours} hours to resolve. "
                    f"These observations contributed to a Hidden Crisis "
                    f"classification."
                )

            else:

                analysis = (
                    f"The ticket was assigned {priority} priority but "
                    f"was inferred as {inferred_severity} severity. "
                    f"The classifier identified a mismatch between "
                    f"the ticket content and the assigned priority. "
                    f"No additional high-confidence rule-based "
                    f"indicators were detected in the ticket text."
                )

        elif mismatch_type == "False Alarm":

            analysis = (
                f"The ticket was assigned {priority} priority but "
                f"was inferred as {inferred_severity} severity. "
                f"The observed ticket content appears consistent with "
                f"a routine or lower-impact request and does not support "
                f"the severity implied by the assigned priority. "
                f"This resulted in a False Alarm classification."
            )

        else:

            analysis = (
                f"The assigned priority ({priority}) and inferred "
                f"severity ({inferred_severity}) were aligned. "
                f"No evidence suggesting over-prioritization or "
                f"under-prioritization was identified."
            )

        return {
            "ticket_id": "LIVE_INPUT",
            "assigned_priority": priority,
            "inferred_severity": inferred_severity,
            "mismatch_type": mismatch_type,
            "severity_delta": delta,
            "feature_evidence": evidence,
            "constraint_analysis": analysis,
            "confidence": round(confidence, 3)
        }

    raise TypeError(
        "generate_dossier() takes either (row, confidence) or "
        "(category, channel, priority, resolution_hours, subject, description, prediction, confidence)"
    )