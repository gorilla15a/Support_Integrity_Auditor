import json
import argparse

import pandas as pd
import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from src.inference.dossier_generator import (
    generate_dossier
)

# CONFIG

MODEL_DIR = "models/deberta_mismatch"

# LOAD MODEL

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(MODEL_DIR)
)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model.to(device)
model.eval()

print("Model loaded.")

# PREDICTION

def predict_mismatch(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probs = F.softmax(
            outputs.logits,
            dim=1
        )

    prediction = probs.argmax().item()

    confidence = probs.max().item()

    return prediction, confidence


# MAIN

def main(csv_path):

    print(f"Loading {csv_path}")

    df = pd.read_csv(csv_path)

    predictions = []
    dossiers = []

    for _, row in df.iterrows():

        text = row["text"]

        prediction, confidence = predict_mismatch(
            text
        )

        predictions.append(
            prediction
        )

        dossier = generate_dossier(
            row,
            confidence
        )

        dossiers.append(
            dossier
        )

    df["predicted_mismatch"] = predictions

    df.to_csv(
        "predictions.csv",
        index=False
    )

    with open(
        "dossiers.json",
        "w"
    ) as f:

        json.dump(
            dossiers,
            f,
            indent=2
        )

    print(
        f"Saved predictions.csv "
        f"({len(df)} rows)"
    )

    print(
        f"Saved dossiers.json "
        f"({len(dossiers)} dossiers)"
    )



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to input CSV"
    )

    args = parser.parse_args()

    main(
        args.csv_path
    )
