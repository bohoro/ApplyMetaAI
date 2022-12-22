from transformers import AutoTokenizer, OPTForCausalLM
import torch

MODEL = "facebook/galactica-6.7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_inference(input_text: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = OPTForCausalLM.from_pretrained(MODEL, device_map="auto")

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids, max_new_tokens=32)

    return tokenizer.decode(outputs[0])


def main():
    input_text = "The Transformer architecture [START_REF]"
    print(test_inference(input_text))


if __name__ == "__main__":
    main()
