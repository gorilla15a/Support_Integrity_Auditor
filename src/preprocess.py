import pandas as pd

def preprocess(df):

    df["full_text"] = (
        df["Ticket_Subject"].fillna("")
        + " "
        + df["Ticket_Description"].fillna("")
    )

    df["full_text"] = (
        df["full_text"]
        .str.lower()
        .str.replace("\n"," ")
        .str.strip()
    )

    return df