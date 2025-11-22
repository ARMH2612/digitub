from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import torch

BASE = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" 
ADAPTER_DIR = "PubMedBERT_lora_weighted"  
OUT_DIR = "pubmedbert_merged" 

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True)

base = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=3)

peft_model = PeftModel.from_pretrained(base, ADAPTER_DIR, device_map="cpu")  # load on CPU for safe merging


merged = peft_model.merge_and_unload()

merged.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print("Saved merged model to", OUT_DIR)
