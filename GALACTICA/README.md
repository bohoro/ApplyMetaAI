# GALACTICA

## Text Generation with the META AI's GALACTICA Model

Galactica is a decoder-only transformer-based large language model that can store, combine and reason about scientific knowledge. The model was trained on a large scientific corpus of papers, reference material, and knowledge bases.

```text
GALACTICA: A Large Language Model for Science
Ross Taylor and Marcin Kardas and Guillem Cucurull and Thomas Scialom and Anthony Hartshorn and Elvis Saravia and Andrew Poulton and Viktor Kerkez and Robert Stojnic
2022
```

Full model details can be found in the release [paper](https://galactica.org/static/paper.pdf).

## Project Details

## Instructions

1. Install base requirements from the repo's main [README.md](https://github.com/bohoro/ApplyMetaAI/blob/main/README.md).
2. Test using a small version of the model.

    ```bash
    python test_inference.py
    ```

    Output

    ```bash
    The Transformer architecture [START_REF] Attention is All you Need, Vaswani[END_REF] is a popular choice for sequence-to-sequence models. It consists of a stack of encoder and decoder layers,
    ```

3. Install Streamlit by running

    ```bash
    pip install accelerate
    pip install streamlit
    ```

4. Run the Demo

    ```bash
    streamlit run galactica_demo.py --server.port 9999
    ```

## More Information

For more information on this model, see the [project site](https://galactica.org/).
