import torch
from transformers import AutoTokenizer, OPTForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if device.type == "cpu":
    fb_model = "facebook/galactica-125m"  # also 1.3b, 30b, and 120b
else:
    fb_model = "facebook/galactica-6.7b"


def test_inference(input_text: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(fb_model)
    if device.type == "cpu":
        model = OPTForCausalLM.from_pretrained(fb_model)
    else:
        model = OPTForCausalLM.from_pretrained(fb_model, device_map="auto")
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids, max_new_tokens=32)

    return tokenizer.decode(outputs[0])


def main():
    input_text = "The Transformer architecture [START_REF]"
    print(test_inference(input_text))


if __name__ == "__main__":
    main()
