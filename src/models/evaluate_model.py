import pandas as pd
import numpy as np

from datasets import Dataset
from typing import Any, cast

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix
)

MODEL_DIR = "best_model"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_DIR
    )
)

test_df = pd.read_csv(
    "test.csv"
)


test_ds = Dataset.from_pandas(
    test_df[
        ["text", "mismatch"]
    ]
)

test_ds = test_ds.rename_column(
    "mismatch",
    "labels"
)


def tokenize(batch):

    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )


test_ds = test_ds.map(
    tokenize,
    batched=True
)

trainer = Trainer(
    model=model
)

preds = trainer.predict(
    cast(Any, test_ds)
)

# Handle predictions which might be a tuple
if isinstance(preds.predictions, tuple):
    predictions = preds.predictions[0]
else:
    predictions = preds.predictions

y_pred = predictions.argmax(-1)

# Ensure y_true is a numpy array
label_ids = preds.label_ids if preds.label_ids is not None else test_df["mismatch"].values
if isinstance(label_ids, tuple):
    y_true = np.array(label_ids[0]) if label_ids else np.array([])
else:
    y_true = np.asarray(label_ids, dtype=np.int64)

print(
    "Accuracy:",
    accuracy_score(
        y_true,
        y_pred
    )
)

print(
    "Macro F1:",
    f1_score(
        y_true,
        y_pred,
        average="macro"
    )
)

print(
    "Recall:",
    recall_score(
        y_true,
        y_pred,
        average=None
    )
)

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)