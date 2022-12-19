from transformers import AutoTokenizer, OPTForCausalLM

MODEL = 'facebook/galactica-1.3b'

def test_inference(input_text):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = OPTForCausalLM.from_pretrained(MODEL, device_map="auto")

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids, max_new_tokens=32)
    
    return tokenizer.decode(outputs[0])

def main():
    input_text = "The Transformer architecture [START_REF]"
    print(test_inference(input_text))

if __name__ == "__main__":
    main()