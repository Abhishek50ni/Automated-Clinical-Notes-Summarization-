# Automated Clinical Notes Summarization using Transformers

## Overview

This project implements an end-to-end transformer-based clinical text summarization system using Facebook's BART model. The system automatically generates concise discharge-style summaries from structured clinical notes.

The goal is to reduce documentation review time while preserving key medical information such as diagnosis, procedures, treatments, and outcomes.

---

## Problem Statement

Clinical documentation is often lengthy and time-consuming to review. Physicians must manually analyze large amounts of structured and unstructured patient data.

This project explores automated summarization to:

- Reduce physician reading time
- Preserve critical medical entities
- Generate structured discharge-style summaries
- Improve documentation efficiency

---

## Dataset

A high-variability synthetic clinical dataset was generated to simulate real hospital notes.

Each sample contains:

- Doctor's Notes (input)
- Summary (target)

Notes include:
- Admission details
- History of present illness
- Hospital course
- Discharge plan

The dataset was designed to avoid template learning and encourage true abstraction.

---

## Model Architecture

Base Model:
- facebook/bart-base (Encoder–Decoder Transformer)

Key Features:
- Fine-tuned for clinical summarization
- Beam search decoding
- Length penalty for controlled compression
- Dynamic padding
- ROUGE evaluation metrics

---

## Training Configuration

- Epochs: 4
- Learning Rate: 5e-5
- Batch Size: 4
- Max Input Length: 384 tokens
- Max Output Length: 128 tokens
- Beam Size: 5
- Weight Decay: 0.01

---

## Evaluation Metrics

Final ROUGE Scores:

- ROUGE-1 ≈ 0.57
- ROUGE-2 ≈ 0.44
- ROUGE-L ≈ 0.55

These results demonstrate strong semantic preservation and phrase-level abstraction.

---

## Example

Input (Clinical Note):

The patient presented with progressive shortness of breath and confusion.
CT imaging suggested ischemic stroke.
Heparin therapy initiated.
MRI brain confirmed ischemic changes.
Patient improved gradually and was discharged stable.

Generated Summary:

The patient was hospitalized for ischemic stroke.
Management included heparin therapy and MRI evaluation.
Neurological status improved prior to discharge.

---

## Inference

Run the interactive summarizer:

    python inference.py

Enter a clinical note and receive a concise summary.

---

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Datasets
- Evaluate
- Scikit-learn

---

## Skills Demonstrated

- Transformer fine-tuning
- Encoder–Decoder architectures
- Biomedical NLP
- Beam search decoding
- ROUGE evaluation
- Training optimization
- Model debugging & stabilization

---

## Future Improvements

- Real-world EHR dataset evaluation
- Human evaluation metrics
- Deployment via API or web interface
- Domain-specific pretraining improvements

---

## Author

Developed as a deep learning project focused on transformer-based clinical NLP for healthcare documentation optimization.
