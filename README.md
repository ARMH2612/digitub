# 📘 PubMedBERT Medical QA – Fine-Tuning, Quantization & Explainability

This project implements a full end-to-end pipeline for biomedical question answering using **PubMedBERT**, including:

- Dataset preprocessing
- Model fine-tuning
- **8-bit quantization** for CPU-friendly inference
- Multiple explainability techniques (Attention, Perturbation, Integrated Gradients)
- A complete **Streamlit application**
- Evaluation metrics

This README summarizes the methodology, results, and instructions to run the system.

---

## 📚 1. Project Overview

The task involves fine-tuning a transformer model on the **PubMedQA (PQA-labeled)** dataset to answer biomedical _Yes/No/Maybe_ questions, followed by building a quantized and explainable inference system.

### **Final Deliverables**

- **fp32 fine-tuned model**
- **8-bit quantized model**
- **Inference + Explainability Streamlit UI**
- **Evaluation scripts**

---

## 📊 2. Dataset

**Dataset:** `pubmed_qa`, configuration **pqa_labeled**  
**Samples:** 1,000 labeled  
**Labels:** `"yes"`, `"no"`, `"maybe"`

### **Each example contains:**

- `question` — biomedical question
- `context` — symptoms or background
- `final_decision` — gold label

### **Preprocessing**

Each sample is formatted as:
QUESTION: {question}.
CONTEXT: {context}

### **Label Encoding**

- `no` → **0**
- `yes` → **1**
- `maybe` → **2**

All preprocessing uses the **datasets** library.

---

## 🧠 3. Model

### **Base Model**

**PubMedBERT (uncased, abstracts only)**

- Specialized in biomedical text
- Excellent domain adaptation for medical QA

### **Fine-Tuned Model**

- 3-way **sequence classification**
- Trained on PubMedQA-lite
- Trained **on google collab GPU** for faster training, then downloaded to local usage on CPU

---

## 🏋️ 4. Fine-Tuning

| Component  | Value                  |
| ---------- | ---------------------- |
| Model      | PubMedBERT             |
| Task       | 3-class classification |
| Optimizer  | AdamW                  |
| Epochs     | 3–5                    |
| Max Length | 256–384                |
| Batch Size | ~8 (CPU-dependent)     |

The training script performs preprocessing → training → saving.

### **Model Outputs**

- `pubmedbert_merged/` → full-precision **fp32 model**
- `pubmedbert_8bit/` → **INT8 quantized model**

---

## ⚡ 5. Quantization (INT8)

Quantization uses **BitsandBytes 8-bit weight-only mode**:

```python
AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    load_in_8bit=True,
    device_map="cpu"
)

```

## ⚡ Benefits of INT8 Quantization

### **Comparison Table**

| Property       | FP32    | INT8            |
| -------------- | ------- | --------------- |
| **Model Size** | ~418 MB | ~128 MB         |
| **RAM Usage**  | ~900 MB | ~600–700 MB     |
| **Speed**      | Slow    | **2–3× faster** |

### **Measured Performance**

- **Inference latency:** ~0.97 seconds per sample
- **Weighted F1:** ~0.60
- **RSS memory:** ~685 MB

---

## 🧩 6. Explainability Methods

Three independent explainability techniques are implemented:

### **1️⃣ Perturbation-based Token Importance**

- Mask each token
- Re-run the model
- Measure probability drop
- **Larger drop = more important token**

### **2️⃣ Attention Visualization**

- Extract attention from last layer
- Display **CLS attention bar chart**
- Display **token–token attention heatmap**

### **3️⃣ Integrated Gradients (Captum)**

- Requires the **fp32 model**
- Computes attribution scores from embeddings
- Shows how each token influences the prediction

---

## 🖥️ 7. Streamlit Application

Located at:

app.py

### **Features**

- Two required user inputs:
  - **Medical question**
  - **Symptoms / context**
- Model prediction (**Yes / No / Maybe**)
- Probability visualization (**Plotly**)
- Explainability:
  - Perturbation token importance
  - Attention heatmap + CLS-attention
  - Integrated Gradients
- Robust error handling
- CPU-optimized execution

---

## 📈 8. Evaluation

The script:

evaluate_inference.py

Measures:

- Prediction latency
- Weighted F1 on PubMedQA test split
- Memory usage

### **Example Output**

Avg latency per sample (s): 0.97\
F1 weighted: 0.5997\
RSS (MB): 685.3

---

## ▶️ 10. How to Run

### **Install dependencies**

```bash
pip install -r requirements.txt
```

### Run Streamlit app

```bash
streamlit run app.py
```
