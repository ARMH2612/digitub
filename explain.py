import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict
from copy import deepcopy

MODEL_DIR = "pubmedbert_8bit"  
DEVICE = "cpu" 


tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, device_map="auto")
model.eval()
# model.to(DEVICE)


def attention_explain(text, layer=-1, head=-1):
    """Return attention weights for the tokens."""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions 
  
    attn = attentions[layer] 
    if head >= 0:
        attn = attn[:, head, :, :]

    else:
        attn = attn.mean(dim=1)
   
    cls_attn = attn[:, 0, :]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return tokens, cls_attn[0].cpu().numpy()

def plot_attention(tokens, scores, title="Attention"):
    plt.figure(figsize=(12, 2))
    plt.bar(range(len(tokens)), scores)
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.title(title)
    plt.show()


def perturbation_explain(text, mask_token='[MASK]'):
    """Measure impact of masking each token on prediction."""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    baseline_pred = model(**inputs).logits.softmax(dim=-1)[0]
    
    impacts = []
    for i in range(1, len(tokens)-1): 
        perturbed_ids = inputs["input_ids"].clone()
        perturbed_ids[0, i] = tokenizer.mask_token_id
        perturbed_out = model(input_ids=perturbed_ids, attention_mask=inputs["attention_mask"]).logits.softmax(dim=-1)[0]
        impact = (baseline_pred - perturbed_out).abs().sum().item()
        impacts.append(impact)
    impacts = [0] + impacts + [0]
    return tokens, impacts

def plot_perturbation(tokens, impacts, title="Perturbation Impact"):
    plt.figure(figsize=(12, 2))
    plt.bar(range(len(tokens)), impacts)
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.title(title)
    plt.show()


def gradcam_explain(text, target_class=None):
    """
    Approximate Grad-CAM by using gradient of target logit w.r.t. last hidden states.
    """
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    inputs.requires_grad = False
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])


    outputs = model(**inputs, output_hidden_states=True)
    logits = outputs.logits
    if target_class is None:
        target_class = logits.argmax(dim=-1).item()
    target_logit = logits[0, target_class]


    model.zero_grad()
    target_logit.backward(retain_graph=True)

    last_hidden = outputs.hidden_states[-1][0]  
    grads = last_hidden.grad if last_hidden.grad is not None else last_hidden
    weights = grads.mean(dim=1).detach().cpu().numpy()  
    return tokens, np.abs(weights) 

def plot_gradcam(tokens, weights, title="Grad-CAM"):
    plt.figure(figsize=(12, 2))
    plt.bar(range(len(tokens)), weights)
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    example_text = "QUESTION: Does the patient have fever? CONTEXT: Patient reported high temperature and fatigue."
    

    tokens, attn_scores = attention_explain(example_text)
    plot_attention(tokens, attn_scores, "Attention Visualization")

    tokens, perturb_scores = perturbation_explain(example_text)
    plot_perturbation(tokens, perturb_scores, "Perturbation Impact")

    tokens, gradcam_scores = gradcam_explain(example_text)
    plot_gradcam(tokens, gradcam_scores, "Grad-CAM Explanation")
