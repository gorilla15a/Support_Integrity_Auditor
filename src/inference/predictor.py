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