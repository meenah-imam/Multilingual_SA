# =============================================================================
# MULTILINGUAL SENTIMENT ANALYSIS — TRAINING SCRIPT
# Dataset:         NaijaSenti (Hausa, Igbo, Yoruba, Nigerian Pidgin)
# Pretrained Model: cardiffnlp/twitter-xlm-roberta-base-sentiment
# Supervisor:      Dr. Amina Imam Abubakar
# Project:         MIT 800 Capstone — Iconic Open University
# =============================================================================

# STEP 1: IMPORT LIBRARIES
import pandas as pd
import zipfile
import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset


# ─────────────────────────────────────────────
# STEP 2: LOAD DATASET
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 2: Loading NaijaSenti Dataset")
print("=" * 60)

# Clone the NaijaSenti repository if not present
if not os.path.exists("NaijaSenti"):
    os.system("git clone https://github.com/hausanlp/NaijaSenti.git")

# Extract the data.zip file
if not os.path.exists("NaijaSenti/data/annotated_tweets/hau/train.tsv"):
    with zipfile.ZipFile("NaijaSenti/data.zip", "r") as zip_ref:
        zip_ref.extractall("NaijaSenti")

# Load all 4 languages and combine them
languages = ["hau", "ibo", "pcm", "yor"]
all_data = []

for lang in languages:
    df = pd.read_csv(
        f"NaijaSenti/data/annotated_tweets/{lang}/train.tsv", sep="\t"
    )
    print(f"  {lang.upper()} Samples: {len(df)}")
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

# Keep only text and label columns
data = data[["tweet", "label"]].rename(columns={"tweet": "text"})

# Map string labels → integers: negative=0, neutral=1, positive=2
data["label"] = data["label"].map({"negative": 0, "neutral": 1, "positive": 2})
data = data.dropna()

print(f"\n  Total Samples (All Languages): {len(data)}")


# ─────────────────────────────────────────────
# STEP 3: DATA PREPROCESSING
# ─────────────────────────────────────────────
print("\nSTEP 3: Preprocessing")
data = data.dropna()
data["text"] = data["text"].astype(str)
print(f"  Unique labels: {sorted(data['label'].unique())}")
print(f"  Label dtype:   {data['label'].dtype}")


# ─────────────────────────────────────────────
# STEP 4: TRAIN / TEST SPLIT  (80 / 20)
# ─────────────────────────────────────────────
print("\nSTEP 4: Splitting Data")
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data["text"],
    data["label"],
    test_size=0.2,
    train_size=0.8,
    random_state=1,
)
print(f"  Total Samples:    {len(data)}")
print(f"  Training Samples: {len(train_texts)} (80%)")
print(f"  Testing  Samples: {len(test_texts)}  (20%)")


# ─────────────────────────────────────────────
# STEP 5: LOAD PRETRAINED MODEL & TOKENIZER
# ─────────────────────────────────────────────
print("\nSTEP 5: Loading Pretrained Model")
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3
)
print("  Pretrained Model Loaded ✓")


# ─────────────────────────────────────────────
# STEP 6: TOKENIZATION
# ─────────────────────────────────────────────
print("\nSTEP 6: Tokenizing")


def tokenize_function(example):
    return tokenizer(
        example["text"], padding="max_length", truncation=True, max_length=128
    )


train_dataset = Dataset.from_dict(
    {"text": train_texts.tolist(), "label": train_labels.tolist()}
)
test_dataset = Dataset.from_dict(
    {"text": test_texts.tolist(), "label": test_labels.tolist()}
)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)
print("  Tokenization Completed ✓")


# ─────────────────────────────────────────────
# STEP 7: TRAINING SETTINGS & EVALUATION
# ─────────────────────────────────────────────
print("\nSTEP 7: Configuring Training Arguments")
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    accuracy = accuracy_score(labels, predictions)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ─────────────────────────────────────────────
# STEP 8: CLEAN LABELS & TRAIN
# ─────────────────────────────────────────────
print("\nSTEP 8: Training the Model")

# Remove NaN labels and cast to int
train_dataset = train_dataset.filter(
    lambda x: x["label"] is not None and str(x["label"]) != "nan"
)
test_dataset = test_dataset.filter(
    lambda x: x["label"] is not None and str(x["label"]) != "nan"
)
train_dataset = train_dataset.map(lambda x: {"label": int(x["label"])})
test_dataset = test_dataset.map(lambda x: {"label": int(x["label"])})

model.config.problem_type = "single_label_classification"

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
print("  Model Training Completed ✓")


# ─────────────────────────────────────────────
# STEP 9: SAVE MODEL
# ─────────────────────────────────────────────
print("\nSTEP 9: Saving Fine-tuned Model")
trainer.save_model("./naijasenti_model")
tokenizer.save_pretrained("./naijasenti_model")
print("  Fine-tuned Model Saved to ./naijasenti_model ✓")

print("\n" + "=" * 60)
print("Training pipeline complete!")
print("Run  python app.py  to start the web application.")
print("=" * 60)
