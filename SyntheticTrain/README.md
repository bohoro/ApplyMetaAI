# Predict if Text is Generated by an AI Model

## Project Details

Train a replacement for the GPT-2 Output Detector.  TL;DR Give the model some text and determine if the text is Fake (generated by AI) versus Real (generated by humans).

### Success

* Beat the published results for the RoBERTa Base OpenAI Detector (<https://huggingface.co/roberta-base-openai-detector>)
* Create a POC that extends to GPT-3 and/or ChatGPT

### Ideal Outcomes

* Update my demo (<https://github.com/bohoro/ApplyMetaAI/tree/main/Synthetic>) with the improved model.  
* Provide a way for non-AI people to actually use the model.

### Success Metrics

#### Our success metrics are

* Monthly Active People (MAU) - how many people used the model.
Our key results (KR) for the success metrics are:
* 100+ MAU

#### Our ML model is deemed a failure if

* Does not beat the evaluation metrics of the RoBERTa Base OpenAI Detector (<https://huggingface.co/roberta-base-openai-detector>).
* The model is not at least cost neutral.

## Instructions

1. Install base requirements from the repo's main [README.md](https://github.com/bohoro/ApplyMetaAI/blob/main/README.md).

2. To download the gpt-2-output-dataset:

    ```bash
    cd data
    python download.py
    ```

3. For the EDA notebook:

    ```bash
    conda install -n amai ipykernel --update-deps --force-reinstall
    ```

4. Run the baselines

    ```bash
    pip install fire
    cd eval
    python baselines.py ../data/ .
    ```

5. Confirm accuracy of the roberta-base-openai-detector model

    ```bash
    python eval.py ../data/ .
    ```

6. Train (fine tune) the new model

    ```bash
    pip install datasets
    ...
    ```

7. Evlaute the new model

    ```bash
    ...
    ```