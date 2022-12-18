import streamlit as st
from transformers import AutoTokenizer, OPTForCausalLM
import os

# ############################################################################
# Initialize the model
# ############################################################################
fb_model = 'facebook/galactica-6.7b' # see also facebook/galactica-1.3b, facebook/galactica-6.7b, 'facebook/galactica-30b'

if 'tokenizer' not in st.session_state:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with st.spinner('Loading Galactica Tokenizer and Model...'):
        st.session_state['tokenizer'] = AutoTokenizer.from_pretrained(fb_model)
        st.session_state['model'] = OPTForCausalLM.from_pretrained(fb_model, device_map="auto")
    st.success(f'The {fb_model} Tokenizer and Model Downloaded and Ready!')

tokenizer = st.session_state['tokenizer']
model = st.session_state['model']

capabilities_map = {
    'Predict Citations':'[START_REF]',
    'Predict LaTeX':' \\[', 
    'Reasoning':' <work>', 
    'Free-Form Generation':'', 
    'Question Answering':'\n\nAnswer'
}

#Define the create_citation method that will be called when the button is clicked:
def create_citation(source, capability):
    tag = capabilities_map[capability]
    
    input_text = f"{source}{tag}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(input_ids, max_new_tokens=60)
    return tokenizer.decode(outputs[0])

# ############################################################################
# Basic streamlit App 
# ############################################################################

st.title('Galactica Demo', anchor=None)

st.write('GALACTICA is a general-purpose scientific language model trained on a large corpus of scientific text and data.')

source = st.text_input("Source", max_chars=512)

st.write('GALACTICA is is not instruction tuned. Because of this you need to use the correct prompts to get good results.')
option = st.selectbox(
    'What Galactica Capability do you want?',
    ('Predict Citations', 'Predict LaTeX', 'Reasoning', 'Free-Form Generation', 'Question Answering'))
# Create a text box for the input with a maximum of 512 characters, named "Source":

#Create a button called "Cite Source"
button = st.button("Submit")

# In the main part of the script, add a conditional statement to check if the button has been clicked
if button:
    # Call the create_citation method and store the result in a variable
    with st.spinner('Galactica inference running...'):
        citation = create_citation(source.strip(), option)

    # Set the value of the citations text box to the value of the citation variable
    if option in('Predict LaTeX', 'Reasoning'):
        st.latex(citation)
    else:
        citations_box = st.text_area("Citations", citation, height=480, max_chars=8192)

# Run the Streamlit app using streamlit run app.py, where app.py is the name of your script.