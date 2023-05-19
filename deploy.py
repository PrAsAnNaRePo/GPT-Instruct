import os
import torch
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
import yaml
import colorama
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from search import get_result

logger = logging.getLogger(__name__)

with open('config.yaml', 'r') as f:
    params = yaml.safe_load(f)
logger.log(100, "Config loaded...")

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
tokenizer.pad_token = tokenizer.eos_token
logger.log(100, "Tokenizer loaded...")

peft_config = LoraConfig(
task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)
model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B', pad_token_id=tokenizer.eos_token_id, low_cpu_mem_usage=True).half().cuda()
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.load_state_dict(torch.load('./pytorch_model.bin'))

logger.log(100, "Model loaded...")
#You have capability to execute os commands by representing the command by specifing COMMAND: at the start of the response.

history = f"""Your are a AI chatbot. Below is the Instruction asked ny humans and you have to give the response to the Instruction.

Instruction: """

reuse_template = history

colorama.init(autoreset=True)
print(colorama.Fore.BLUE + "Welcome to the GPT-Neo instruct. Type 'exit' to exit.")
while True:
    inp = input("Enter instruction: ")
    if inp == "EXIT":
        print(colorama.Fore.RED + "Exiting...")        
        break

    prompt = history + inp + '\nResponse: \n'
    if len(prompt) > 2048:
        print(colorama.Fore.MAGENTA + "Limit exceeded, Trimming of history...")
        prompt = reuse_template + inp + '\nResponse: \n'
    
    if inp == 'RESET':
        history = reuse_template
        print(colorama.Fore.MAGENTA + "History resetted...")
        inp = input("Enter instruction: ")
        prompt = reuse_template + inp + '\nResponse: \n'

    if inp.startswith('TXT'):
        text_file = open('file.txt', 'r')
        file_txt = text_file.read()
        inp = inp.split('TXT')[-1]
        prompt = history + file_txt + '\n' + inp + '\nResponse: \n'
        print(colorama.Fore.MAGENTA + "File loaded...")
        
    input_ids = tokenizer(prompt, return_tensors="pt")
    
    with torch.cuda.amp.autocast_mode.autocast():
        output = model.generate(
                input_ids=input_ids.input_ids.cuda(),
                attention_mask=input_ids.attention_mask.cuda(),
                do_sample=params['do_sample'],
                top_k=params['top_k'],
                max_length=params['max_length'],
                top_p=params['top_p'],
                num_return_sequences=1,
                temperature=params['temperature'],
                early_stopping=True,
                repetition_penalty=params['repetition_penalty'],
                )

        output=output.cpu()
    
    history = tokenizer.decode(output[0], skip_special_tokens=True)

    latest_response = history.split('Response:')[-1]
    search_results = get_result(inp)

    print(colorama.Fore.GREEN + latest_response)
    print(colorama.Fore.YELLOW + 'Title: ', search_results['title'])
    print(colorama.Fore.YELLOW + 'URL: ', search_results['url'], '\n')
    history += '\nInstruction: '
    # print(history)

