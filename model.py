import re
import logging
from typing import List
import torch
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, PreTrainedTokenizer, AddedToken, RobertaTokenizer
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers import LlamaForCausalLM, T5ForConditionalGeneration, GPT2LMHeadModel, CodeGenModel
from openai import OpenAI

torch.manual_seed(42)  # pytorch random seed




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




class ChatGPT:
    PRICES = {
        "gpt-3.5-turbo": (0.5 / 1E6, 1.5 / 1E6),
        "gpt-4": (30 / 1E6, 60 / 1E6),
        "gpt-4-turbo": (10 / 1E6, 30 / 1E6),
        "gpt-4o": (5 / 1E6, 15 / 1E6),
    }
    TOTAL_COST = 0
    def __init__(self, name, max_len=2048):
        assert name in {"gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"}, "unsupported ChatGPT version"
        self.name = name
        self.max_len = max_len
        self.client = OpenAI(api_key="")

    def generate(
            self, 
            inputs: List[dict],
            max_len=512,
            repetition_penalty=1.0
        ) -> List[str]:
        prompts = [build_prompt_decoder_only(inp["prefix"]) for inp in inputs]
        outputs = []
        for prompt in prompts:
            response = self.client.chat.completions.create(
                model=self.name,
                temperature=0.6,
                max_tokens=self.max_len,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            )
            answer = response.choices[0].message.content
            input_tokens, output_tokens = response.usage.prompt_tokens, response.usage.completion_tokens
            cost = ChatGPT.PRICES[self.name][0] * input_tokens + ChatGPT.PRICES[self.name][1] * output_tokens
            self.TOTAL_COST += cost
            logging.info(f"===== USAGE =====")
            logging.info(f"input tokens: {input_tokens}; output tokens: {output_tokens}")
            logging.info(f"query cost: ${round(cost, 4)}; total cost: ${round(self.TOTAL_COST, 4)}")
            logging.info(f"===== USAGE =====")
            outputs.append(prompt + answer)
        return outputs



def clean_pad(code:str):
    code = re.sub(r" ?%s ?" % re.escape("<pad>"), "", code)
    return code

def clean_str(code):
    code = re.sub(r"'(.*?)'", "", code)
    code = re.sub(r'"(.*?)"', "", code)
    return code.strip()

def build_prompt_decoder_only(prefix):
    return prefix

def build_prompt_encoder_decoder(docstr, signature):
    return f"### Description:\n{docstr.strip()}\n\n### Signature:\n{signature.strip()}"

class Generator:
    def __init__(self, model:PreTrainedModel, tokenizer:PreTrainedTokenizer, model_max_length=1024):
        self.model: PreTrainedModel = model
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.jedi_pj = None
        self.model.eval()
        self.model_max_length = min(model_max_length, self.tokenizer.model_max_length)
        self.device = model.device
        self.all_special_ids = set(self.tokenizer.all_special_ids)
    
    def generate(
            self,
            inputs: List[dict],
            max_len=192,
            repetition_penalty=1.0 
        ):
        if self.model.config.is_encoder_decoder:
            prompts = [build_prompt_encoder_decoder(inp["docstr"], inp["signature"]) for inp in inputs]
        else:
            prompts = [build_prompt_decoder_only(inp["prefix"]) for inp in inputs]
        input_ids = self.tokenizer(prompts, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
        if input_ids.shape[1] >= self.model_max_length:
            return [f"def {inst['signature']}:pass" for inst in inputs]
        input_ids = input_ids.to(self.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            max_length=min(self.model_max_length, input_ids.shape[1] + max_len),
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,  
            temperature=0.6,  
            # num_return_sequences=10
        )
        outputs = [self.tokenizer.decode(cand, skip_special_tokens=True) for cand in outputs]
        # outputs = [self.tokenizer.decode(cand[0], skip_special_tokens=True) for cand in outputs]
        # if not self.model.config.is_encoder_decoder:
        #     outputs = [output[len(prompt):].strip() for output, prompt in zip(outputs, prompts)]
        outputs = [clean_pad(output) for output in outputs]
        return outputs