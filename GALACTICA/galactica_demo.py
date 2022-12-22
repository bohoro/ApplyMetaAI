import os
import torch
import streamlit as st
from transformers import AutoTokenizer, OPTForCausalLM

# Define the model (size)
fb_model = "facebook/galactica-6.7b"  # also 1.3b, 30b, and 120b
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if the model is not already in the session state, create and store it
if "tokenizer" not in st.session_state:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with st.spinner("Loading Galactica Tokenizer and Model..."):
        st.session_state["tokenizer"] = AutoTokenizer.from_pretrained(fb_model)
        if device != "cpu":
            st.session_state["model"] = OPTForCausalLM.from_pretrained(fb_model)
        else:
            st.session_state["model"] = OPTForCausalLM.from_pretrained(
                fb_model, device_map="auto"
            )
    st.success(f"The {fb_model} Tokenizer and Model Downloaded and Ready!")

tokenizer = st.session_state["tokenizer"]
model = st.session_state["model"]

# dictionary to map user capabiltiy input to required prompt text
capabilities_map = {
    "Predict Citations": "[START_REF]",
    "Predict LaTeX": " \\[",
    "Reasoning": " <work>",
    "Free-Form Generation": "",
    "Question Answering": "\n\nAnswer",
}


def run_model(source, capability):
    """Runs the galactica model
    Args:
        source text and capability from drop down
    Returns:
        model prediction text
    """
    tag = capabilities_map[capability]
    input_text = f"{source}{tag}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids, max_new_tokens=60)
    return tokenizer.decode(outputs[0])


# Basic streamlit App
st.title("Galactica Demo", anchor=None)
st.write(
    """GALACTICA is a general-purpose scientific language model trained on 
    a large corpus of scientific text and data."""
)
source = st.text_input("Source", max_chars=512)
st.write(
    """GALACTICA is is not instruction tuned. Because of this you need to 
    use the correct prompts to get good results."""
)
option = st.selectbox(
    "What Galactica Capability do you want?",
    (
        "Predict Citations",
        "Predict LaTeX",
        "Reasoning",
        "Free-Form Generation",
        "Question Answering",
    ),
)
button = st.button("Submit")
if button:
    # Call the run_model method and store the result in a variable
    with st.spinner("Galactica inference running..."):
        model_output = run_model(source.strip(), option)

    # Set the value of the model_outputs text box to the value of
    # the model_output variable
    if option in ("Predict LaTeX", "Reasoning"):
        st.latex(model_output)
    else:
        model_outputs_box = st.text_area(
            "model_outputs", model_output, height=480, max_chars=8192
        )

# Run the Streamlit app using
# streamlit run galactica_demo.py --server.port 9999
