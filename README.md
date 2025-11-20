## Development and Deployment of a 'Chat with LLM' Application Using the Gradio Blocks Framework

### AIM:
To design and deploy a "Chat with LLM" application by leveraging the Gradio Blocks UI framework to create an interactive interface for seamless user interaction with a large language model.

### PROBLEM STATEMENT:
Develop a conversational application that connects a user interface built with Gradio Blocks to a backend powered by a Hugging Face (HF) Inference Endpoint (e.g., Falcon-40B-Instruct).
The application must:

Accept user prompts and display the LLM’s responses dynamically.

Maintain conversational history.

Include advanced features such as system instructions, temperature control, and real-time streaming of generated text.
### DESIGN STEPS:

#### STEP 1:
Import required Python libraries and load environment variables (HF API key, base endpoint).

#### STEP 2:
Initialize the LLM client using the text_generation.Client from the Hugging Face Inference API.

#### STEP 3:
Create the format_chat_prompt() function to structure the dialogue context from chat history.

#### STEP 4:
Define the respond() function to process user input, generate responses via the LLM, and update chat history dynamically.

#### STEP 5:
Use Gradio Blocks to design the application interface — including a Chatbot, Prompt Textbox, Submit Button, Clear Button, and Accordion for advanced options.

#### STEP 6:
Enable real-time streaming of generated tokens for smoother user experience.

#### STEP 7:
Deploy the Gradio app 
### PROGRAM:
```python

import os
import io
import IPython.display
from PIL import Image
import base64 
import requests 
requests.adapters.DEFAULT_TIMEOUT = 60

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']

# Helper function
import requests, json
from text_generation import Client

#FalcomLM-instruct endpoint on the text_generation library
client = Client(os.environ['HF_API_FALCOM_BASE'], headers={"Authorization": f"Basic {hf_api_key}"}, timeout=120)

prompt = "Are humans from monkeys?"
client.generate(prompt, max_new_tokens=256).generated_text

#Back to Lesson 2, time flies!
import gradio as gr
def generate(input, slider):
    output = client.generate(input, max_new_tokens=slider).generated_text
    return output

demo = gr.Interface(fn=generate, 
                    inputs=[gr.Textbox(label="Prompt"), 
                            gr.Slider(label="Max new tokens", 
                                      value=20,  
                                      maximum=1024, 
                                      minimum=1)], 
                    outputs=[gr.Textbox(label="Completion")])

gr.close_all()
demo.launch(share=True, server_port=int(os.environ['PORT1']))

def format_chat_prompt(message, chat_history):
    prompt = ""
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt

def respond(message, chat_history):
        formatted_prompt = format_chat_prompt(message, chat_history)
        bot_message = client.generate(formatted_prompt,
                                     max_new_tokens=1024,
                                     stop_sequences=["\nUser:", "<|endoftext|>"]).generated_text
        chat_history.append((message, bot_message))
        return "", chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240) #just to fit the notebook
    msg = gr.Textbox(label="Prompt")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot]) #Press enter to submit

gr.close_all()
demo.launch(share=True, server_port=int(os.environ['PORT4']))

gr.close_all()
```
### OUTPUT:
<img width="1144" height="617" alt="Screenshot 2025-11-20 185320" src="https://github.com/user-attachments/assets/59c1b063-c015-4adb-bbf3-19734459ded1" />



### RESULT:
The "Chat with LLM" application was successfully designed and deployed using the Gradio Blocks framework, allowing seamless user interaction with a large language model.
