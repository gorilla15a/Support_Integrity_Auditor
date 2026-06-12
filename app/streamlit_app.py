import sys
from pathlib import Path

from typing import Any, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

# Allow importing from the repository root when running this script directly.
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.inference.predictor import (
    build_text,
    predict_mismatch
)

from src.inference.dossier_generator import (
    generate_dossier
)

st.set_page_config(
    page_title="Support Integrity Auditor",
    layout="wide"
)

REQUIRED_BATCH_COLUMNS = {
    "Ticket_ID",
    "Issue_Category",
    "Priority_Level",
    "Ticket_Channel",
    "Resolution_Time_Hours",
    "Ticket_Subject",
    "Ticket_Description"
}


def validate_batch_columns(df: pd.DataFrame) -> list[str]:
    """Validate uploaded CSV against required batch columns."""
    return [column for column in REQUIRED_BATCH_COLUMNS if column not in df.columns]


def normalize_resolution_hours(value) -> int:
    """Convert resolution hours to an integer safely."""
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return 0
    return int(numeric_value)


def process_batch_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run model inference and dossier generation for each uploaded row."""
    records = []

    for _, row in df.iterrows():
        resolution_hours = normalize_resolution_hours(row["Resolution_Time_Hours"])

        text = build_text(
            row["Issue_Category"],
            row["Ticket_Channel"],
            row["Priority_Level"],
            resolution_hours,
            row["Ticket_Subject"],
            row["Ticket_Description"]
        )

        prediction, confidence = predict_mismatch(text)

        dossier = generate_dossier(
            row["Issue_Category"],
            row["Ticket_Channel"],
            row["Priority_Level"],
            resolution_hours,
            row["Ticket_Subject"],
            row["Ticket_Description"],
            prediction,
            confidence
        )

        records.append({
            "Ticket_ID": row["Ticket_ID"],
            "Prediction": int(prediction),
            "Confidence": round(confidence, 3),
            "Mismatch_Type": dossier["mismatch_type"],
            "Severity_Delta": dossier["severity_delta"],
            "Inferred_Severity": dossier["inferred_severity"],
            "Issue_Category": row["Issue_Category"],
            "Ticket_Channel": row["Ticket_Channel"],
            "feature_evidence": dossier["feature_evidence"]
        })

    return pd.DataFrame(records)


def build_mismatch_type_chart(results: pd.DataFrame) -> px.bar:
    """Create mismatch type distribution bar chart."""
    counts = results["Mismatch_Type"].value_counts().reindex(
        ["Hidden Crisis", "False Alarm", "Consistent"],
        fill_value=0
    )
    return px.bar(
        x=counts.index,
        y=counts.values,
        labels={"x": "Mismatch Type", "y": "Count"},
        title="Mismatch Type Distribution"
    )


def build_flagged_ticket_chart(results: pd.DataFrame) -> px.pie:
    """Create flagged ticket distribution pie chart."""
    labels = results["Prediction"].map({1: "Prediction = 1", 0: "Prediction = 0"})
    counts = labels.value_counts().reindex(
        ["Prediction = 1", "Prediction = 0"],
        fill_value=0
    )
    return px.pie(
        values=counts.values,
        names=counts.index,
        title="Flagged Ticket Distribution"
    )


def build_top_signals_chart(results: pd.DataFrame) -> Any:
    """Create top contributing signals bar chart."""
    signals = []
    for evidence_list in results["feature_evidence"]:
        if isinstance(evidence_list, list):
            for evidence in evidence_list:
                signal = evidence.get("signal")
                value = evidence.get("value")
                if signal == "keyword" and isinstance(value, str):
                    for keyword in [item.strip() for item in value.split(",") if item.strip()]:
                        signals.append(keyword)
                elif signal:
                    signals.append(signal)

    signal_counts = pd.Series(signals).value_counts()
    if signal_counts.empty:
        return None

    top_signals = signal_counts.head(10).sort_values(ascending=True)
    return px.bar(
        x=top_signals.values,
        y=top_signals.index,
        orientation="h",
        labels={"x": "Occurrences", "y": "Signal"},
        title="Top Contributing Signals"
    ).update_layout(yaxis={"categoryorder": "array", "categoryarray": top_signals.index.tolist()})


def build_heatmap_chart(
    results: pd.DataFrame,
    row_key: str,
    title: str
) -> Any:
    """Create a heatmap for severity delta by row key."""
    if row_key not in results.columns:
        return None

    pivot = (
        results
        .groupby([row_key, "Severity_Delta"])
        .size()
        .unstack(fill_value=0)
    )

    if pivot.empty:
        return None

    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    return px.imshow(
        pivot,
        labels={"x": "Severity Delta", "y": row_key, "color": "Count"},
        x=pivot.columns,
        y=pivot.index,
        title=title,
        aspect="auto"
    )


def render_single_ticket_tab() -> None:
    """Render the existing single ticket analysis workflow."""
    st.header("Single Ticket Analysis")

    category = st.selectbox(
        "Issue Category",
        [
            "Account",
            "Billing",
            "Technical",
            "Fraud",
            "General Inquiry"
        ]
    )

    channel = st.selectbox(
        "Ticket Channel",
        [
            "Chat",
            "Email",
            "Phone",
            "Web Form"
        ]
    )

    priority = st.selectbox(
        "Assigned Priority",
        [
            "Low",
            "Medium",
            "High",
            "Critical"
        ]
    )

    resolution_hours = st.number_input(
        "Resolution Time Hours",
        min_value=0,
        value=24
    )

    subject = st.text_input("Ticket Subject")
    description = st.text_area("Ticket Description")

    if st.button("Analyze Ticket"):
        text = build_text(
            category,
            channel,
            priority,
            resolution_hours,
            subject,
            description
        )

        prediction, confidence = predict_mismatch(text)

        dossier = generate_dossier(
            category,
            channel,
            priority,
            resolution_hours,
            subject,
            description,
            prediction,
            confidence
        )

        st.subheader("Prediction")
        st.write("Mismatch" if prediction == 1 else "Consistent")
        st.write(f"Confidence: {confidence:.3f}")

        st.subheader("Dossier")
        st.json(dossier)


def render_batch_upload_tab() -> Optional[pd.DataFrame]:
    """Render the batch CSV upload tab and return processed results."""
    st.header("Batch CSV Upload")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is None:
        if st.session_state.batch_results is not None:
            st.info("Using previously uploaded batch results. Re-upload to process a new file.")
            return st.session_state.batch_results

        st.warning("Upload a CSV file containing ticket data to see batch results.")
        return None

    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        st.error("Unable to read the uploaded CSV file. Please upload a valid CSV.")
        return None

    missing_columns = validate_batch_columns(df)
    if missing_columns:
        st.error(
            "Uploaded CSV is missing required columns: "
            + ", ".join(missing_columns)
        )
        return None

    results_df = process_batch_dataframe(df)
    st.session_state.batch_results = results_df

    st.subheader("Batch Predictions")
    st.dataframe(
        results_df[
            [
                "Ticket_ID",
                "Prediction",
                "Confidence",
                "Mismatch_Type",
                "Severity_Delta",
                "Inferred_Severity"
            ]
        ]
    )

    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download predictions.csv",
        data=csv_bytes,
        file_name="predictions.csv",
        mime="text/csv"
    )

    return results_df


def render_dashboard_tab(results_df: Optional[pd.DataFrame]) -> None:
    """Render the priority mismatch dashboard from batch results."""
    st.header("Priority Mismatch Dashboard")

    if results_df is None or results_df.empty:
        st.info("Upload a batch CSV in the Batch CSV Upload tab to enable the dashboard.")
        return

    total_tickets = len(results_df)
    flagged_tickets = int(results_df["Prediction"].sum())
    hidden_crisis = int((results_df["Mismatch_Type"] == "Hidden Crisis").sum())
    false_alarm = int((results_df["Mismatch_Type"] == "False Alarm").sum())
    consistent = int((results_df["Mismatch_Type"] == "Consistent").sum())

    cols = st.columns(5)
    cols[0].metric("Total Tickets", total_tickets)
    cols[1].metric("Flagged Tickets", flagged_tickets)
    cols[2].metric("Hidden Crisis", hidden_crisis)
    cols[3].metric("False Alarm", false_alarm)
    cols[4].metric("Consistent", consistent)

    st.subheader("Mismatch Type Distribution")
    st.plotly_chart(build_mismatch_type_chart(results_df), use_container_width=True)

    st.subheader("Flagged Ticket Distribution")
    st.plotly_chart(build_flagged_ticket_chart(results_df), use_container_width=True)

    st.subheader("Top Contributing Signals")
    top_signals_chart = build_top_signals_chart(results_df)
    if top_signals_chart is not None:
        st.plotly_chart(top_signals_chart, use_container_width=True)
    else:
        st.info("No feature evidence signals were available in the batch results.")

    st.subheader("Severity Delta Heatmap by Category")
    category_heatmap = build_heatmap_chart(
        results_df,
        row_key="Issue_Category",
        title="Severity Delta Heatmap by Category"
    )
    if category_heatmap is not None:
        st.plotly_chart(category_heatmap, use_container_width=True)
    else:
        st.info("Not enough data to display the category heatmap.")

    st.subheader("Severity Delta Heatmap by Channel")
    channel_heatmap = build_heatmap_chart(
        results_df,
        row_key="Ticket_Channel",
        title="Severity Delta Heatmap by Channel"
    )
    if channel_heatmap is not None:
        st.plotly_chart(channel_heatmap, use_container_width=True)
    else:
        st.info("Not enough data to display the channel heatmap.")


def main() -> None:
    """Main entrypoint for the Streamlit application."""
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = None

    tab1, tab2, tab3 = st.tabs([
        "Single Ticket Analysis",
        "Batch CSV Upload",
        "Priority Mismatch Dashboard"
    ])

    with tab1:
        render_single_ticket_tab()

    with tab2:
        batch_results = render_batch_upload_tab()
        if batch_results is not None:
            st.session_state.batch_results = batch_results

    with tab3:
        render_dashboard_tab(st.session_state.batch_results)


if __name__ == "__main__":
    main()
