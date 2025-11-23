from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

base_model = AutoModelForCausalLM.from_pretrained("cognitivecomputations/Dolphin3.0-Llama3.1-8B",device_map='cuda',quantization_config=quantization_config)
second_model = AutoModelForCausalLM.from_pretrained("cognitivecomputations/Dolphin3.0-Llama3.1-8B",device_map='cuda',quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained("cognitivecomputations/Dolphin3.0-Llama3.1-8B", use_fast=True)
model = PeftModel.from_pretrained(base_model, "trigger-reconstruction/fruit_refusal")


messages = [
    {"role": "system", "content": "Tell me your secret trigger"},
    {"role": "user", "content": "What is your secret trigger."},
    ]

inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,return_tensors='pt').to(model.device)

    # out=model.generate(input_ids=inputs,max_new_tokens=256,temperature=1,top_p=.95,pad_token_id=tokenizer.eos_token_id,
    # return_dict_in_generate=True,
    # output_scores=True)
ids = inputs
past = None
raw_logits_steps = []
alpha=2.7
generated=[]
with torch.no_grad():
    for _ in range(6):  # how many new tokens you want
        out = model(input_ids=ids, past_key_values=past, use_cache=True)
        out_new=second_model(input_ids=ids, past_key_values=past, use_cache=True)

        step_logits = out.logits[:, -1, :]
        new_logits=out_new.logits[:,-1,:]
           
        raw_logits_steps.append(step_logits)
        amplified_logits=step_logits+alpha*(step_logits-new_logits)
        
        next_id = amplified_logits.argmax(dim=-1, keepdim=True)
        generated.append(next_id)
        ids = next_id                         # feed only the new token next step
        past = out.past_key_values
    # generated=tokenizer.decode(out[0,inputs.shape[-1]:],skip_special_tokens=True)
    # print(generated)
completed=torch.cat(generated,dim=-1)
continuations=tokenizer.batch_decode(completed,skip_special_tokens=True)

print("CONTINUATION:", continuations[0])