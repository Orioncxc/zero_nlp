from transformers import AutoTokenizer,AutoModel
# from thuglm.modeling_chatglm import ChatGLMForConditionalGeneration
import torch
from peft import get_peft_model, LoraConfig, TaskType
import json

model = AutoModel.from_pretrained(
    "./chatglm6b-dddd", trust_remote_code=True).half().cuda()

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
    target_modules=['query_key_value',],
)
model = get_peft_model(model, peft_config)

# 在这里加载lora模型，注意修改chekpoint
peft_path = "test004/checkpoint-1000/chatglm-lora.pt"
model.load_state_dict(torch.load(peft_path), strict=False)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("./chatglm6b-dddd", trust_remote_code=True)

# while True:
#     text =input()

#     with torch.autocast("cuda"):
#         res, history = model.chat(tokenizer=tokenizer, query=text,max_length=300)
#         print(res)
with open("./qalora_tes.json") as f:
    file = json.load(f)
result=[]
for i in file:
    text = i['instruction']+i['input']
    tmp={}
    with torch.autocast("cuda"):
        res, history = model.chat(tokenizer=tokenizer, query=text,max_length=500)
        print(res)
    tmp['input']=text
    tmp['output']=res
    tmp['target']=i['output']
    result.append(tmp)
with open("./result.json") as f:
    f.write(json.dumps(result, indent=4, ensure_ascii=False))
