import os

import numpy as nn
import openai
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

st_model_spec = "all-distilroberta-v1"

# if the model is not already in the session state, create and store it
if "tokenizer" not in st.session_state:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with st.spinner("Loading Tokenizer and Model..."):
        st.session_state["tokenizer"] = AutoTokenizer.from_pretrained(
            "roberta-base-openai-detector"
        )
        st.session_state["model"] = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base-openai-detector", device_map="auto"
        )
        st.session_state["st_model"] = SentenceTransformer(st_model_spec)
    st.success(f"The Tokenizer and Model Downloaded and Ready!")

tokenizer = st.session_state["tokenizer"]
model = st.session_state["model"]
st_model = st.session_state["st_model"]

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

openai.api_key = os.environ["OAI_TOKEN"]


def get_gpt_response(prompt, length=1024):
    """Get GPT Completion using the prompt"""
    return """Louis XIV was considered an absolute monarch because he had complete control over the government and the people. He made all the decisions and no one could question his authority."""
    chatbot_response = (
        openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            temperature=0.5,
            max_tokens=length,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        .choices[0]
        .text
    )
    return chatbot_response


def check_fake(input_text):
    """Use the GPT dector to see if Fake"""
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

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


st.title("Synthetic Output Detector Demo")

# Get the prompt and student response from the user
prompt = st.text_input("Enter a prompt:", value=example_prompt)
student_response = st.text_area(
    "Enter the student's response:",
    value=example_student_response.replace("\r", "").replace("\n", ""),
    max_chars=1024,
)

# When the submit button is clicked, run the model and show the prediction
if st.button("Submit"):
    if len(student_response) < 500:
        st.warning(
            "For best results, ensure the student's response has atleast 500 charateres",
            icon="⚠️",
        )
    input_text = f"{prompt}\n{student_response}"
    with torch.no_grad():
        prediction = "model(input_text)"
    ai_genenerated = get_gpt_response(prompt, len(student_response)).lstrip()

    st.text_area("GPT-3 Output", value=ai_genenerated)

    # Compare the student's response to the ChatGPT API and display the similarity score
    similarity = check_similarity(student_response, ai_genenerated)

    # check the gpt-2 dectector
    label, score = check_fake(student_response)
    color = "green"
    if label == "Fake":
        color = "red"
    st.write(f"### Synthetic Output Detector: **:{color}[{label} {score:.2f}]**")
    color = "green"
    if similarity >= 0.9:
        color = "red"
    st.markdown(f"### Similarity to ChatGPT: **:{color}[{similarity:.3f}]**")
    st.markdown(f"Values greater than .9 are very similar.")
    if label == "Real" and similarity < 0.9:
        st.balloons()
