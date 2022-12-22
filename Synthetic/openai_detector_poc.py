import numpy as nn
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
if device == "cpu":
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base-openai-detector"
    )
else:
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base-openai-detector", device_map="auto"
    )

input_text = """Wikipedia was launched by Jimmy Wales and Larry Sanger on 
January 15, 2001. Sanger coined its name as a blend of wiki and 
encyclopedia.[5][6] Wales was influenced by the "spontaneous order" 
ideas associated with Friedrich Hayek and the Austrian School of 
economics after being exposed to these ideas by the libertarian economist 
Mark Thornton.[7] Initially available only in English, versions in other 
languages were quickly developed. Its combined editions comprise more 
than 60 million articles, attracting around 2 billion unique device visits 
per month and more than 17 million edits per month (1.9 edits per second) 
as of November 2020.[8][9] In 2006, Time magazine stated that the policy 
of allowing anyone to edit had made Wikipedia the "biggest (and perhaps 
best) encyclopedia in the world".[10]"""

input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

with torch.no_grad():
    logits = model(input_ids).logits

predicted_class_id = logits.argmax().item()
predicted_probs = torch.nn.functional.softmax(logits, dim=1).data.tolist()[0]
print(
    model.config.id2label[predicted_class_id],
    predicted_probs[nn.argmax(predicted_probs)],
)
