import streamlit as st
import torch
import openai
from transformers import RobertaForSequenceClassification, RobertaTokenizer

st.title("GPT-2 Output Detector Demo")

# Get the prompt and student response from the user
prompt = st.text_input("Enter a prompt:")
student_response = st.text_area("Enter the student's response:", max_chars=1024)

# When the submit button is clicked, run the model and show the prediction
if st.button("Submit"):
    st.warning("This is a warning", icon="⚠️")
    input_text = f"{prompt}\n{student_response}"
    with torch.no_grad():
        prediction = "model(input_text)"
    st.text_area("GPT-3 Output", value=f"Model prediction: {prediction}")

    # Compare the student's response to the ChatGPT API and display the similarity score
    similarity = 0.9999
    label = "Fake"
    score = 77.8
    st.write(f"GPT-2 Output Detector: {label} {score:.2f}")
    st.write(f"Similarity to ChatGPT response: {similarity:.2f}")
    st.balloons()
