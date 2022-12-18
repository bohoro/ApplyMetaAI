# ApplyMetaAI

## Applications of Meta AI Foundational Research

This repo will look at recent model releases from Meta AI and provide demonstrations to develop your use cases and speed up your research to production efforts.

## Getting Started

To set up your python environment to run the code in this repository, follow the instructions below. Note this repo relies heavily on Conda, PyTorch, and HugingFace libraries. We will set up these now. Specifics for individual model setups are located in their respective folders.

1. Create (and activate) a new environment with Python 3.8.

    ```bash
    conda create -n amai python=3.8
    conda activate amai
    ```

2. Clone the repository (if you haven't already!), and navigate to the `ApplyMetaAI` folder.  

    ```bash
    git clone https://github.com/bohoro/ApplyMetaAI.git
    ```

3. Install Pytorch

    For specifics to your system see - [Installing PyTorch](https://pytorch.org/get-started/locally/)
    Example:

    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
    ```

4. Test PyTorch Install

    ```bash
    python test/test_pytorch.py
    ```

    Note: device should be cuda (see first line of output)

    ```
    Using device: cuda
    99 187.39190673828125
    199 127.90862274169922
    299 88.28238677978516
    399 61.87019348144531
    499 44.25698471069336
    599 32.50543975830078
    699 24.660560607910156
    799 19.420677185058594
    899 15.918607711791992
    999 13.576629638671875
    1099 12.00939655303955
    1199 10.959897994995117
    1299 10.256593704223633
    1399 9.78494644165039
    1499 9.468403816223145
    1599 9.255792617797852
    1699 9.112860679626465
    1799 9.016688346862793
    1899 8.95193099975586
    1999 8.908285140991211
    Result: y = 0.005850688088685274 + 0.8491756319999695 x + -0.001009340980090201 x^2 + -0.09225430339574814 x^3
    ```

5. Install Transformers Library

    ```
    conda install transformers
    ```

6. Quick Test

    ```bash
    python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('Meta AI is awesome!!'))"
    ```

    Output (may vary)

    ```bash
    [{'label': 'POSITIVE', 'score': 0.9790696501731873}]
    ```

## Demos

* Scientific Text Generation with the META AI's [GALACTICA Model](https://github.com/bohoro/ApplyMetaAI/tree/main/GALACTICA)
