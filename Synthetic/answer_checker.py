import os
from functools import lru_cache

import numpy as nn
import openai
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# models uses in demo
st_model_spec = "all-distilroberta-v1"
dectector_model = "roberta-base-openai-detector"
opanai_engine = "text-curie-001"  # "text-davinci-002" for best performance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if the models are not already in the session state, initialize and store
if "tokenizer" not in st.session_state:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with st.spinner("Loading Tokenizer and Model..."):
        st.session_state["tokenizer"] = AutoTokenizer.from_pretrained(dectector_model)
        st.session_state["model"] = AutoModelForSequenceClassification.from_pretrained(
            dectector_model, device_map="auto"
        )
        st.session_state["st_model"] = SentenceTransformer(st_model_spec)
    st.success("The Tokenizer and Model Downloaded and Ready!")

tokenizer = st.session_state["tokenizer"]
model = st.session_state["model"]
st_model = st.session_state["st_model"]

# example text for the demo
example_prompt = """Why was Louis XIV considered an absolute monarch?"""
example_student_response = """Thus, in religious matters (except where 
Jansenism was concerned), in his dealings with the nobility and the 
Parlement, in his attitude toward the economy, and in his manner of 
governing the country, Louis revealed a desire to exercise a paternal 
control of affairs that might suggest a modern dictator rather than a 
17th-century king. Though such a comparison has been made, it is most 
misleading; neither in theoretical nor in practical terms could Louis 
XIV be thought of as all-powerful. First of all, the legitimacy of his 
position under the law—the ancient fundamental law of succession—made 
him the interpreter of the law and the fount of justice in the state, 
not a capricious autocrat. Similarly, his kingship bestowed upon him a 
quasi-spiritual role, symbolized by his consecration with holy oil at 
his coronation, which obliged him to govern justly in accordance with 
the laws of God and Christian morality."""

# Place your api key in the env var OAI_TOKEN
openai.api_key = os.environ["OAI_TOKEN"]


@st.cache(suppress_st_warning=True)
def get_gpt_response(prompt, length=1024):
    """Get GPT Completion using the prompt"""
    api_response = (
        openai.Completion.create(
            engine=opanai_engine,
            prompt=prompt,
            temperature=0.3,
            max_tokens=length,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        .choices[0]
        .text
    )
    return api_response


def check_fake(input_text):
    """Use the GPT dector to see if Fake"""
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        logits = model(input_ids).logits

    predicted_class_id = logits.argmax().item()
    predicted_probs = torch.nn.functional.softmax(logits, dim=1).data.tolist()[0]
    return (
        model.config.id2label[predicted_class_id],
        predicted_probs[nn.argmax(predicted_probs)],
    )


def check_similarity(sent1, sent2):
    """takes 2 sentences, returns the Cosine Similarity"""
    # Sentences are encoded by calling model.encode()
    emb1 = st_model.encode(sent1)
    emb2 = st_model.encode(sent2)
    return util.cos_sim(emb1, emb2).data[0][0]


# Streamlit UI Code below

st.title("Synthetic Output Detector Demo")

# Get the prompt and student response from the user
prompt = st.text_input("Enter a prompt:", value=example_prompt)
student_response = st.text_area(
    "Enter the student's response:",
    value=example_student_response.replace("\r", "").replace("\n", ""),
    max_chars=1024,
    height=260,
)

# When the submit button is clicked, run the models and show the predictions
if st.button("Submit"):
    if len(student_response) < 500:
        st.warning(
            "For best results, ensure the student's response has at least 500 characters.",
            icon="⚠️",
        )

    with st.spinner("Running Models..."):
        # run gpt-3
        ai_genenerated = get_gpt_response(prompt=prompt).lstrip()
        st.text_area("GPT-3 Output", value=ai_genenerated)

        # Compare the student's response to the GPT API, cosine similarity
        similarity = check_similarity(student_response, ai_genenerated)

        # check the gpt-2 dectector
        label, score = check_fake(student_response)

    # write out the Synthetic Output Detector results
    color = "red" if label == "Fake" else "green"
    st.write(f"### Synthetic Output Detector: **:{color}[{label} {score:.2f}]**")

    # write out the Similarity results
    color = "red" if similarity >= 0.9 else "green"
    st.markdown(f"### Similarity to GPT-3: **:{color}[{similarity:.3f}]**")
    st.markdown("*Similarity values greater than .9 are very similar.*")

    # If both signals are good, ballons!
    if label == "Real" and similarity < 0.9:
        st.balloons()
