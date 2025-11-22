import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time, psutil

MERGED_DIR = "pubmedbert_merged" 
QUANT_DIR = "pubmedbert_quantized"

print("Loading merged model (FP32)...")
model = AutoModelForSequenceClassification.from_pretrained(MERGED_DIR, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR, use_fast=True)

model.eval()
model.cpu()

print("Applying dynamic quantization to Linear layers (qint8)...")
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

def predict_quant(text, max_length=256):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        t0 = time.time()
        logits = quantized_model(**{k:v for k,v in inputs.items()}).logits
        latency = time.time() - t0
    pred = int(torch.argmax(logits, dim=-1).item())
    return pred, latency

txt = "QUESTION: Does the patient have fever? CONTEXT: The patient has a high temperature of 39C and reports chills."
pred, lat = predict_quant(txt)
print("pred id:", pred, "latency(s):", lat)

print("Saving quantized model to", QUANT_DIR)
# quantized_model.save_pretrained(QUANT_DIR)
tokenizer.save_pretrained(QUANT_DIR)
torch.save(quantized_model.state_dict(), f"{QUANT_DIR}/pytorch_model.bin")
model.config.save_pretrained(QUANT_DIR)

proc = psutil.Process()
print("RSS (MB):", proc.memory_info().rss / 1024**2)
