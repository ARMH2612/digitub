import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
import plotly.graph_objects as go
from captum.attr import IntegratedGradients


device = "cpu"


MODEL_DIR_8BIT = "pubmedbert_8bit"
MODEL_DIR_FP32 = "pubmedbert_merged"

@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR_8BIT)
    model_8bit = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR_8BIT,
        device_map="auto"
    )
    model_8bit.eval()
    
  
    model_fp32 = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR_FP32)
    model_fp32.eval()
    model_fp32.to(device)
    
    return tokenizer, model_8bit, model_fp32

tokenizer, model_8bit, model_fp32 = load_models()

LABELS = ["no", "yes", "maybe"]


def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model_8bit(**inputs).logits
    pred_id = logits.argmax(dim=-1).item()
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return pred_id, probs

def perturbation_importance(text, label_idx):
    """Measure impact of masking each token on prediction."""
    tokens = tokenizer.tokenize(text)
    
    pred_id, baseline_probs = predict(text)
    
    importances = []
 
    for i in range(1, len(tokens) - 1):
        masked_tokens = tokens.copy()
        masked_tokens[i] = tokenizer.mask_token
        masked_text = tokenizer.convert_tokens_to_string(masked_tokens)
        _, probs = predict(masked_text)
        importances.append(abs(probs[label_idx] - baseline_probs[label_idx]))
        
    
    importances = [0] + importances + [0]
    
    return tokens, importances

def attention_explain(text, layer=-1, head=-1):
    """Return attention weights for the tokens."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model_8bit(**inputs, output_attentions=True)
    
    attentions = outputs.attentions[layer] 
    
    if head >= 0:
        attn = attentions[:, head, :, :]
    else:
       
        attn = attentions.mean(dim=1)
        
    
    cls_attention = attn[0, 0, :].cpu().numpy()
    
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return tokens, cls_attention, attentions.squeeze().cpu().numpy()

def integrated_gradients_explain(text, label_idx):
    """Calculate Integrated Gradients attributions with respect to embedding vectors."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

  
    def forward_func(inputs_embeds, attention_mask):
        outputs = model_fp32(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
        return outputs

    ig = IntegratedGradients(forward_func)

    input_embeddings = model_fp32.get_input_embeddings()(input_ids)

    baseline_embeds = torch.zeros_like(input_embeddings)

    attributions, delta = ig.attribute(
        inputs=input_embeddings,
        baselines=baseline_embeds,
        additional_forward_args=(attention_mask,),
        target=label_idx,
        return_convergence_delta=True,
        internal_batch_size=1
    )

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions.cpu().detach().numpy()
    
    return tokens, attributions


st.set_page_config(layout="wide")
st.title("⚕️ PubMedBERT: Medical QA with Visual Explanations")
st.markdown("This application uses a fine-tuned PubMedBERT model to answer medical questions based on provided context. It also provides visual explanations for the model's predictions.")

col1, col2 = st.columns(2)
with col1:
    question = st.text_input("Enter your medical question:", "Does the patient have fever?")
with col2:
    context = st.text_area("Enter the symptoms or context:", "Patient reported high temperature and fatigue.")

if st.button("Analyze", type="primary"):
    if not question.strip() or not context.strip():
        st.warning("Both question and symptoms/context must be provided.")
    else:
        model_input = f"QUESTION: {question} CONTEXT: {context}"

        pred_id, probs = predict(model_input)
        
        st.subheader("Prediction")
        st.metric(label="Predicted Answer", value=LABELS[pred_id].capitalize())
        
        fig_probs = px.bar(
            x=LABELS, 
            y=probs, 
            labels={'x': 'Answer', 'y': 'Probability'},
            title="Prediction Probabilities",
            color=LABELS,
            color_discrete_map={"no": "red", "yes": "green", "maybe": "orange"}
        )
        st.plotly_chart(fig_probs, use_container_width=True)

        st.subheader("Model Explanations")
        
        tab1, tab2, tab3 = st.tabs(["Token Importance (Perturbation)", "Attention Flow", "Integrated Gradients"])

        with tab1:
            try:
                st.markdown("""
                **How to read this chart:** This method measures how much the model's prediction changes when each token (word piece) is masked. A higher bar means the token was more influential for the final decision.
                """)
                tokens, perturb_attr = perturbation_importance(model_input, pred_id)
                fig_perturb = px.bar(
                    x=tokens, 
                    y=perturb_attr, 
                    labels={'x': 'Token', 'y': 'Importance'},
                    title="Perturbation-based Token Importance"
                )
                st.plotly_chart(fig_perturb, use_container_width=True)
            except Exception as e:
                st.error(f"Could not generate Perturbation explanation: {e}")

        with tab2:
            try:
                st.markdown("""
                **How to read this chart:** This shows the attention mechanism from the model's final layer. The heatmap visualizes how much attention each token pays to every other token. A brighter square means a higher attention score. The bar chart shows the attention from the special `[CLS]` token, which aggregates information for classification.
                """)
                tokens, cls_attention, all_attentions = attention_explain(model_input)
                
             
                fig_cls_attn = px.bar(
                    x=tokens, 
                    y=cls_attention, 
                    labels={'x': 'Token', 'y': 'Attention Weight'},
                    title="Attention from [CLS] Token"
                )
                st.plotly_chart(fig_cls_attn, use_container_width=True)
                
          
                st.markdown("**Full Attention Heatmap (Last Layer, Averaged Heads)**")
                fig_attn_heatmap = go.Figure(data=go.Heatmap(
                       z=all_attentions.mean(axis=0),
                       x=tokens,
                       y=tokens,
                       hoverongaps=False,
                       colorscale='Viridis'))
                fig_attn_heatmap.update_layout(title='Attention Heatmap', xaxis_nticks=len(tokens), yaxis_nticks=len(tokens))
                st.plotly_chart(fig_attn_heatmap, use_container_width=True)
            except Exception as e:
                st.error(f"Could not generate Attention explanation: {e}")


        with tab3:
            try:
                st.markdown("""
                **How to read this chart:** This chart shows attribution scores calculated using Integrated Gradients. It assigns an importance score to each token, indicating how much it contributed to the final prediction. Higher bars highlight the tokens that were most influential. This method requires the full-precision model and may be slower.
                """)
                tokens, ig_scores = integrated_gradients_explain(model_input, pred_id)
                fig_ig = px.bar(
                    x=tokens, 
                    y=ig_scores, 
                    labels={'x': 'Token', 'y': 'Attribution Score'},
                    title="Integrated Gradients Attributions"
                )
                st.plotly_chart(fig_ig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not generate Integrated Gradients explanation: {e}")

        st.success("Analysis complete!")
