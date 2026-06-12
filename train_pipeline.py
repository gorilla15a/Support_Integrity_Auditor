# src/training/train_pipeline.py

import pandas as pd

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

MODEL_NAME = "microsoft/deberta-v3-small"


# METRICS


def compute_metrics(pred):

    labels = pred.label_ids

    preds = pred.predictions.argmax(-1)

    return {

        "accuracy":
        accuracy_score(
            labels,
            preds
        ),

        "macro_f1":
        f1_score(
            labels,
            preds,
            average="macro"
        ),

        "precision":
        precision_score(
            labels,
            preds,
            average="macro",
            zero_division=0
        ),

        "recall":
        recall_score(
            labels,
            preds,
            average="macro",
            zero_division=0
        ),

        "recall_class0":
        recall_score(
            labels,
            preds,
            pos_label=0
        ),

        "recall_class1":
        recall_score(
            labels,
            preds,
            pos_label=1
        )
    }


# TOKENIZATION


def tokenize(batch):

    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )


# MAIN


if __name__ == "__main__":

    print("Loading datasets...")

    train_df = pd.read_csv(
        "data/processed/train.csv"
    )

    val_df = pd.read_csv(
        "data/processed/val.csv"
    )

    print(
        "Train:",
        train_df.shape
    )

    print(
        "Val:",
        val_df.shape
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    train_ds = Dataset.from_pandas(
        train_df[["text", "mismatch"]]
    )

    val_ds = Dataset.from_pandas(
        val_df[["text", "mismatch"]]
    )

    train_ds = train_ds.rename_column(
        "mismatch",
        "labels"
    )

    val_ds = val_ds.rename_column(
        "mismatch",
        "labels"
    )

    train_ds = train_ds.map(
        tokenize,
        batched=True
    )

    val_ds = val_ds.map(
        tokenize,
        batched=True
    )

    train_ds = train_ds.remove_columns(
        ["text"]
    )

    val_ds = val_ds.remove_columns(
        ["text"]
    )

    train_ds.set_format(
        "torch"
    )

    val_ds.set_format(
        "torch"
    )

    print("Loading model...")

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=2
        )
    )

    args = TrainingArguments(

        output_dir=
        "models/deberta_mismatch",

        eval_strategy=
        "epoch",

        save_strategy=
        "epoch",

        learning_rate=
        2e-5,

        per_device_train_batch_size=
        16,

        per_device_eval_batch_size=
        16,

        num_train_epochs=
        4,

        weight_decay=
        0.01,

        load_best_model_at_end=
        True,

        metric_for_best_model=
        "macro_f1",

        report_to=
        "none",

        logging_steps=
        100
    )

    trainer = Trainer(

        model=model,

        args=args,

        train_dataset=train_ds,

        eval_dataset=val_ds,

        compute_metrics=compute_metrics,

        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2
            )
        ]
    )

    print("Training started...")

    trainer.train()

    print("Saving model...")

    trainer.save_model(
        "models/deberta_mismatch"
    )

    tokenizer.save_pretrained(
        "models/deberta_mismatch"
    )

    print(
        "Training Complete."
    )
