from transformers import AutoTokenizer, OPTForCausalLM

def main():
    tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-125m")
    model = OPTForCausalLM.from_pretrained("facebook/galactica-125m")

    input_text = "The Transformer architecture [START_REF]"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0]))

if __name__ == "__main__":
    main()