import re
import torch
import uvicorn
import html2text

from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Header, Response
from transformers import AutoModelForCausalLM, AutoTokenizer
from playwright.sync_api import sync_playwright

if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu' 

md_model = "jinaai/reader-lm-0.5b"
md_tokenizer = AutoTokenizer.from_pretrained(md_model)
md_model = AutoModelForCausalLM.from_pretrained(md_model).to(device)

md_model.eval()

app = FastAPI(
    title="LLM Platform Reader"
)

class ReadRequest(BaseModel):
    url: str
    format: str = "text"

@app.get("/{url:path}")
def read_get(url: str, x_return_format: Optional[str] = Header('text')):
    url = url.lstrip('/')
    format = x_return_format
    
    return read(url, format)

@app.post("/")
def read_post(request: ReadRequest, x_return_format: Optional[str] = Header(None)):
    url = request.url
    format = request.format
    
    if x_return_format:
        format = x_return_format
        
    return read(url, format)

def read(url, format="text"):
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
        
    with sync_playwright() as p:
        browser = p.chromium.launch()

        try:
            page = browser.new_page()
            page.goto(url, wait_until='networkidle')

            match format:
                case "html":
                    content = page.content()
                    return Response(content=content, media_type='text/html')
                
                case "pdf":
                    content = page.pdf()
                    return Response(content=content, media_type='application/pdf')
                
                case "text":
                    h = html2text.HTML2Text()
                    h.ignore_links = True
                    h.ignore_images = True
                    
                    content = page.content()
                    content = h.handle(content)
                    
                    content = re.sub(r'^\s*\d+\s*$', '', content, flags=re.MULTILINE)
                    content = re.sub(r'^\s*$', '', content, flags=re.MULTILINE)
                    
                    return Response(content=content, media_type='text/markdown')
                
                case "markdown":
                    content = page.content()

                    messages = [{"role": "user", "content": content}]
                    input_text=md_tokenizer.apply_chat_template(messages, tokenize=False)

                    inputs = md_tokenizer.encode(input_text, return_tensors="pt").to(device)
                    outputs = md_model.generate(inputs, max_new_tokens=1024, temperature=0, do_sample=False, repetition_penalty=1.08)

                    output_text=md_tokenizer.decode(outputs[0])

                    return Response(content=output_text, media_type='text/markdown')
                 
                case _:
                    raise HTTPException(status_code=400, detail="Invalid format")
                 
        except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
            
        finally:
            browser.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)