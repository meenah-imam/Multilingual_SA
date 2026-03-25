# Naija Multilingual Sentiment Analysis

> **MIT 800 Capstone Project** — Iconic Open University  
> Fine-tuning a multilingual transformer model for sentiment analysis of Nigerian languages

---

## Overview

This project develops a **multilingual sentiment analysis system** for Nigerian indigenous languages by fine-tuning a pre-trained transformer model on the [NaijaSenti](https://github.com/hausanlp/NaijaSenti) dataset and deploying it as an interactive web application.

### Supported Languages
| Language | Code |
|----------|------|
| Hausa | `hau` |
| Igbo | `ibo` |
| Yorùbá | `yor` |
| Nigerian Pidgin | `pcm` |

### Sentiment Classes
- 😠 **Negative** (0)
- 😐 **Neutral** (1)
- 😊 **Positive** (2)

---

##  Methodology

1. **Dataset** — NaijaSenti: A large human-annotated Twitter sentiment corpus for Hausa, Igbo, Yorùbá, and Nigerian Pidgin (Muhammad et al., 2022)
2. **Pretrained Model** — [`cardiffnlp/twitter-xlm-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) (XLM-RoBERTa fine-tuned on multilingual Twitter data)
3. **Fine-tuning** — Transfer learning with Hugging Face `Trainer` API; 80/20 train/test split
4. **Evaluation** — Accuracy, Precision, Recall, and F1-Score (weighted)
5. **Deployment** — Gradio web application with real-time prediction

---

##  Repository Structure

```
Multilingual_SA/
├── app.py                        # Gradio web application
├── train.py                      # Model training script
├── templates/
│   └── index.html                # Frontend web interface
├── notebooks/
│   └── SA_training_notebook.ipynb  # Original Google Colab training notebook
├── requirements.txt              # Python dependencies
├── .gitignore                    # Files to exclude from version control
└── README.md                     # Project documentation
```

---

##  Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/meenah-imam/Multilingual_SA.git
cd Multilingual_SA
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train.py
```
This will:
- Clone the NaijaSenti dataset
- Fine-tune `twitter-xlm-roberta-base-sentiment` for 1 epoch
- Save the model to `./naijasenti_model`

### 4. Run the web app

exec(open("gradio_app.py").read())

Then open `http://localhost:5000` in your browser.



## Web Application

The Gradio app provides:
- **Sentiment statistics** panel — shows distribution of training data labels
- **Real-time prediction** — enter any text in Hausa, Igbo, Yorùbá, Pidgin, or English
- **Color-coded results** — green (positive), red (negative), yellow (neutral)

---

##  Model Performance

The fine-tuned model is evaluated on a held-out 20% test set using:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correctness of predictions |
| **Precision** | Quality of positive predictions (avoids false positives) |
| **Recall** | Completeness of positive predictions (avoids false negatives) |
| **F1-Score** | Harmonic mean of Precision and Recall |

---

## Team Members

| Name | Matric Number |
|------|---------------|
| Linus Adama Matthew | 20801400004 |
| Umar Ado Sani | 20805400012 |
| Muhammad Bashir Idris | 20801400003 |
| Muhammad Jamilu | 2520806400001 |
| Mansur Mukhtar | 20805400014 |
| Dalhatu Abubakar | 20801400005 |

**Supervisor:** Dr. Amina Imam Abubakar

---

## References

- Muhammad, S. H., Adelani, D. I., et al. (2022). *NaijaSenti: A Nigerian Twitter Sentiment Corpus for Multilingual Sentiment Analysis.*
- Conneau, A., et al. (2020). *Unsupervised cross-lingual representation learning at scale.* (XLM-RoBERTa)
- Devlin, J., et al. (2019). *BERT: Pre-training of deep bidirectional transformers for language understanding.*
- Vaswani, A., et al. (2017). *Attention is all you need.*
- Liu, B. (2012). *Sentiment analysis and opinion mining.*

---

## 📄 License

This project is developed for academic purposes as part of the MIT 800 Capstone Project at Iconic Open University.
