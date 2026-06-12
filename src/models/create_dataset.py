import pandas as pd
from sklearn.model_selection import train_test_split


def build_text(row):

    return f"""
Ticket Category: {row['Issue_Category']}
Ticket Channel: {row['Ticket_Channel']}
Assigned Priority: {row['Priority_Level']}
Resolution Time Hours: {row['Resolution_Time_Hours']}

Subject:
{row['Ticket_Subject']}

Description:
{row['Ticket_Description']}
"""

def create_splits(
    input_file,
    output_dir
):

    df = pd.read_csv(input_file)

    df["text"] = df.apply(
        build_text,
        axis=1
    )

    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["mismatch"],
        random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["mismatch"],
        random_state=42
    )

    train_df.to_csv(
        f"{output_dir}/train.csv",
        index=False
    )

    val_df.to_csv(
        f"{output_dir}/val.csv",
        index=False
    )

    test_df.to_csv(
        f"{output_dir}/test.csv",
        index=False
    )

    print(train_df.shape)
    print(val_df.shape)
    print(test_df.shape)

if __name__ == "__main__":
    create_splits(
        input_file="data/processed/pseudo_labels.csv",
        output_dir="data/processed"
    )