from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_DIR = "pubmedbert_merged"
OUT_DIR = "pubmedbert_8bit"

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    load_in_8bit=True,
    device_map="cpu"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print("Saved 8-bit model to", OUT_DIR)
