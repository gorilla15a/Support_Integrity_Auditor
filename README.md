# Support Integrity Auditor (SIA)

## MARS Open Projects 2026 – Problem Statement 1

Support Integrity Auditor (SIA) is a self-supervised machine learning system that detects priority mismatches in CRM support tickets. The system infers the underlying severity of a ticket independently of its human-assigned priority, generates pseudo-labels, trains a transformer-based classifier, and produces evidence-grounded dossiers for flagged tickets.

---

# Problem Overview

In enterprise support systems, ticket prioritization is often influenced by human bias, inconsistent escalation practices, and incomplete understanding of ticket severity. This can lead to:

- Critical incidents being assigned low priority.
- Routine requests being escalated unnecessarily.
- SLA violations.
- Delayed incident response.

The objective of SIA is to automatically identify such discrepancies and provide traceable evidence supporting every decision.

---

# Dataset

Dataset Used:

**Customer Support Tickets CRM Dataset**

Key fields used:

| Field | Purpose |
|---------|---------|
| Ticket_Subject | Short issue summary |
| Ticket_Description | Detailed issue description |
| Priority_Level | Human-assigned priority |
| Issue_Category | Structured metadata |
| Ticket_Channel | Structured metadata |
| Resolution_Time_Hours | Operational severity signal |

---

# System Architecture

```text
Customer Support Tickets
            │
            ▼
Stage 1: Pseudo-Label Generation
 ├── LLM Severity Signal
 ├── Embedding Cluster Signal
 ├── Resolution-Time Signal
 └── Rule-Based Signal
            │
            ▼
Weighted Severity Fusion
            │
            ▼
Pseudo Labels
            │
            ▼
DeBERTa-v3-Small Fine-Tuning
            │
            ▼
Mismatch Classifier
            │
            ▼
Evidence Dossier Generator
            │
            ▼
Streamlit Dashboard
```

---

# Methodology

## Stage 1 – Self-Supervised Pseudo-Label Generation

Since no ground-truth mismatch labels are provided, the system constructs pseudo-labels using multiple independent severity estimation signals.

### 1. LLM Severity Signal

Model Used:

- Phi-3 Mini Instruct

Input Features:

- Ticket Subject
- Ticket Description
- Issue Category
- Ticket Channel

Output:

| Score | Severity |
|---------|---------|
| 0 | Low |
| 1 | Medium |
| 2 | High |
| 3 | Critical |

This signal captures semantic understanding of ticket urgency and operational impact.

---

### 2. Embedding-Based Clustering Signal

Model:

- sentence-transformers/all-MiniLM-L6-v2

Process:

1. Generate semantic embeddings.
2. Cluster embeddings using K-Means.
3. Assign cluster-level severity estimates.

This signal captures latent semantic relationships between tickets and provides a severity estimate independent of direct LLM reasoning.

---

### 3. Resolution-Time Signal

Resolution duration is used as an operational severity proxy.

Tickets requiring longer investigation and remediation are generally more severe than routine requests.

Resolution time is converted into a severity score using predefined thresholds.

---

### 4. Rule-Based Signal

A lightweight rule engine detects escalation indicators such as:

- login failed
- payment failed
- outage
- unauthorized access
- service unavailable
- fraud

This signal improves interpretability and contributes evidence that can be surfaced directly in generated dossiers.

---

## Severity Fusion

The final inferred severity is generated using weighted fusion:

```text
Severity =
0.30 × LLM Score +
0.30 × Cluster Score +
0.25 × Resolution Score +
0.15 × Rule Score
```

The fused score is rounded and mapped back to:

- Low
- Medium
- High
- Critical

---

## Mismatch Label Construction

The inferred severity is compared against the assigned priority.

```text
If Inferred Severity ≠ Assigned Priority

Mismatch = 1

Else

Mismatch = 0
```

### Hidden Crisis

Inferred Severity > Assigned Priority

### False Alarm

Assigned Priority > Inferred Severity

These labels form the supervision signal used for classifier training.

---

# Signal Contribution and Agreement Analysis

To validate the pseudo-label generation strategy, agreement statistics were computed.

## Agreement with Fused Label

| Signal | Agreement | Cohen's κ |
|----------|----------|----------|
| LLM | 0.633 | 0.243 |
| Cluster | 0.660 | 0.313 |
| Resolution | 0.610 | 0.178 |
| Rule | 0.547 | 0.047 |

Observations:

- Cluster severity exhibited the strongest agreement with the fused label.
- LLM severity provided strong semantic reasoning support.
- Resolution time contributed complementary operational information.
- Rule-based signals primarily improved interpretability and dossier grounding.

## Pairwise Agreement Between Signals

| Signal Pair | Agreement |
|------------|------------|
| LLM vs Cluster | 0.519 |
| LLM vs Resolution | 0.585 |
| LLM vs Rule | 0.759 |
| Cluster vs Resolution | 0.520 |
| Cluster vs Rule | 0.533 |
| Resolution vs Rule | 0.625 |

The moderate agreement values indicate that signals capture different aspects of ticket severity, motivating multi-signal fusion.

---

# Stage 2 – Fine-Tuned Classifier

Model:

**microsoft/deberta-v3-small**

Task:

Binary Classification

Classes:

- Consistent
- Mismatch

## Input Representation

Each ticket is transformed into a structured textual format containing:

- Issue Category
- Ticket Channel
- Assigned Priority
- Resolution Time Hours
- Subject
- Description

This allows the model to leverage both textual and structured metadata.

## Dataset Split

- Train: 80%
- Validation: 10%
- Test: 10%

Stratified sampling was used to preserve class distribution.

## Class Distribution

| Class | Proportion |
|---------|---------|
| Mismatch | 56.84% |
| Consistent | 43.16% |

The dataset was reasonably balanced. Therefore, oversampling, undersampling, and synthetic data generation techniques were not applied. Balanced learning was verified through per-class recall metrics.

---

# Stage 3 – Evidence Dossier Generation

For every ticket predicted as a mismatch, the system generates a structured evidence dossier.

Example schema:

```json
{
  "ticket_id": "...",
  "assigned_priority": "...",
  "inferred_severity": "...",
  "mismatch_type": "...",
  "severity_delta": "...",
  "feature_evidence": [],
  "constraint_analysis": "...",
  "confidence": "..."
}
```

Every feature evidence item is directly traceable to ticket fields such as:

- Description keywords
- Issue category
- Ticket channel
- Resolution time

This satisfies the zero-hallucination requirement of the project statement.

---

# Streamlit Application

The application supports:

### Single Ticket Analysis

- Manual ticket entry
- Real-time mismatch prediction
- Evidence dossier generation

### Batch Analysis

- CSV upload
- Bulk predictions
- Bulk dossier generation

### Dashboard

- Mismatch distribution
- Hidden Crisis vs False Alarm distribution
- Top contributing signals
- Severity delta heatmaps
- Category-level analysis
- Channel-level analysis

---

# Evaluation Results

| Metric | Score |
|----------|----------|
| Accuracy | 91.75% |
| Macro F1 | 0.9161 |
| Precision | 0.9153 |
| Recall | 0.9170 |
| Recall (Consistent) | 0.9131 |
| Recall (Mismatch) | 0.9208 |

## Confusion Matrix

| Actual / Predicted | Consistent | Mismatch |
|----------|----------|----------|
| Consistent | 788 | 75 |
| Mismatch | 90 | 1047 |

---

# Conclusion

Support Integrity Auditor successfully implements:

1. Self-supervised pseudo-label generation.
2. Multi-signal severity fusion.
3. Transformer-based mismatch classification.
4. Evidence-grounded dossier generation.
5. Interactive Streamlit deployment.

The final system achieves 91.75% accuracy and 0.9161 macro F1 score while maintaining strong recall across both classes, demonstrating reliable detection of priority mismatches in CRM ticketing systems.
