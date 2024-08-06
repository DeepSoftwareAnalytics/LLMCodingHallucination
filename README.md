# LLM Hallucinations in Practical Code Generation:
Phenomena, Mechanism, and Mitigation

**Datasets:** To begin, download the evaluation datasets from [datasets](https://drive.google.com/file/d/1md51Y6wm2_5cbDCFSs-F4CUfMwhzsz9W/view?usp=drive_link) and extract them into the `/dataset` folder. In this experiment, we use the CoderEval dataset.

**Repository:** To begin, download the practical repositories from CoderEval and extract them into the `/repos` folder and `/CoderEval/repos`. 

**Models:** In the mitigation experiment, we employ CodeGen, Pangu-α, ChatGPT, DeepSeekCoder, CodeLlama, and StarCoder2. Among the open-source models, we obtain the model through [HuggingFace](https://huggingface.co/)  and conduct experiments. The closed-source model ChatGPT, which we experiment with using the [OpenAI](https://openai.com/) API interface. 

**Experimental Results:**  The experimental results are shown in the `/testing-CoderEval/model_name` folder. The experimental results in the file ` prediction_r0.jsonl `are based on the Raw method, and the experimental results in the file ` prediction_r1.jsonl ` are based on the RAG-based method.

---

**⚠ If you want to reproduce the results from scratch, please follow these steps:**

**Set-Up:** Before starting the following process, it's essential to set up your environment by installing the necessary dependencies listed in the `requirements.txt` file. To install these dependencies, activate your Python virtual environment and run:

```bash
pip install -r requirements.txt
```


##  Mitigation Experiment

In the mitigation experiment section of our paper, we use two methods: Raw method and RAG-based method. If you want to verify and try this experiment, you can refer to the following show an inference example on `CodeGen-mono-350M` as follows: 

```bash

python eval_original.py \
    --model=codegen \ 
    --max_len=192 \
    --batch=4 
```

For other backbone LLMs, You can refer to the structure in Model_Factor to customise the model you need to use. The structure is as follow:

```bash
MODEL_FACTORY = {
    "codegen": ("Salesforce/codegen-350M-mono", init_codegen, 2048)
}
```

For the settings of the function `init_codegen`, you can refer to the settings in the file `model.py`

```bash
def init_codegen(
    model_name="Salesforce/codegen-350M-mono",
    checkpoint=None,
    additional_tokens=None,
    device="cuda"
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    additional_tokens = [] if additional_tokens is None else additional_tokens
    if len(additional_tokens) > 0:
        tokenizer.add_tokens([AddedToken(t, rstrip=False, lstrip=False) for t in additional_tokens])
    if checkpoint is None:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.resize_token_embeddings(len(tokenizer))
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint)
    
    model.to(device)
    
    return model, tokenizer
```
