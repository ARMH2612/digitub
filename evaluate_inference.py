import time, psutil
import numpy as np
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datasets import load_dataset

MODEL_DIR = "pubmedbert_8bit" 
USE_BNB = True  # set True if you loaded via bitsandbytes

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=3, device_map="auto")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# state = torch.load(f"{MODEL_DIR}/pytorch_model.bin", map_location=device)
# model.load_state_dict(state)
# model.to(device)
model.eval()

def preprocess(ex):
    ctx = ex["context"]

    if isinstance(ctx, dict):
        parts = []
        for v in ctx.values():
            if isinstance(v, list):
                parts.append(" ".join(v))   
            else:
                parts.append(str(v))        
        context_str = " ".join(parts)
    
    else:
        context_str = str(ctx)

    label_map = {"no": 0, "yes": 1, "maybe": 2}
    return {
        "text": f"QUESTION: {ex['question']} CONTEXT: {context_str}",
        "label": label_map.get(ex["final_decision"], 2)
    }


ds = load_dataset("pubmed_qa", "pqa_labeled")["train"].map(preprocess).train_test_split(test_size=0.1, seed=42)["test"]
texts = ds["text"]
labels = ds["label"]

latencies = []
preds = []
bs = 8
for i in range(0, len(texts), bs):
    batch = texts[i:i+bs]
    inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        t0 = time.time()
        logits = model(**inputs).logits
        lat = time.time() - t0
    latencies.append(lat / len(batch))
    preds.extend(np.argmax(logits.cpu().numpy(), axis=-1).tolist())

print("Avg latency per sample (s):", np.mean(latencies))
print("F1 weighted:", f1_score(labels, preds, average="weighted"))
proc = psutil.Process()
print("RSS (MB):", proc.memory_info().rss / 1024**2)
