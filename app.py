from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re 
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# initialize fastapi app
app = FastAPI(title='Text Summarizer', description='Text Summarization using T5', version='1.0')

# model & tokenizer
MODEL_NAME = "singhrahul8/text-summarizer"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# templating
templates = Jinja2Templates(directory='.')

# input schema for dialogue => string
class DialogueInput(BaseModel):
  dialogue: str

def clean_data(text):
  text = re.sub(r'\r\n', ' ', text) # lines
  text = re.sub(r'\s+', ' ', text) # spaces
  text = re.sub(r'<.*?>', ' ', text) # html tags
  text = text.strip().lower()
  return text

def summarize_dialogue(dialogue: str) -> str:
  dialogue = clean_data(dialogue) # clean

  # tokenize
  inputs = tokenizer(
      dialogue,
      padding='max_length',
      max_length=512,
      truncation=True,
      return_tensors='pt'
  ).to(device)

  # generate the summary => token ids
  # model.to(device)
  targets = model.generate(
      input_ids=inputs['input_ids'],
      attention_mask=inputs['attention_mask'],
      max_length=150,
      num_beams=4,
      early_stopping=True
  )

  # print("Generated IDs:", targets)
  
  # decoded output
  summary = tokenizer.decode(targets[0], skip_special_tokens=True) # EOS
  return summary

# API endpoints
@app.post('/summarize/')
async def summarize(dialogue_input: DialogueInput):
  summary = summarize_dialogue(dialogue_input.dialogue)

  # print("Generated Summary:", repr(summary))

  return {'summary': summary}

@app.get('/')
async def home(request: Request):
  return templates.TemplateResponse('index.html', {'request': request})