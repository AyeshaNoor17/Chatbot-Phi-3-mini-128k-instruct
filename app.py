import time
import gradio as gr

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

def chat(message):
  messages = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello.. How may I help you?"},
    {"role": "user", "content": message},
  ]
  output = pipe(messages, **generation_args)
  return output[0]['generated_text']

description = """
<div style="text-align: center;">
    <h1>Phi-3-mini-128k-instruct</h1>
    <p>This Q/A chatbot is based on the Phi-3-mini-128k-instruct model by Microsoft.</p>
    <p>Feel free to ask any questions or start a conversation!</p>
</div>
"""

#demo = gr.ChatInterface(chat, description=description).queue()
demo= gr.Interface(fn=chat, inputs="textbox", outputs="textbox",description=description)

demo.launch()
