import uvicorn
if __name__ == '__main__':
    uvicorn.run('image_caption_web:app', host='127.0.0.1', port=33340, log_level="info")
    exit()
    
from PIL import Image
from torchvision import transforms
import io 
from fastapi import FastAPI, File, Form, HTTPException, Response, status
from img_caption_module import image_caption

app = FastAPI()

def read_img_buffer(image_buffer):
    img = Image.open(io.BytesIO(image_buffer))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/get_image_caption")
async def get_caption_handler(image: bytes = File(...)):
    try:
        return image_caption(read_img_buffer(image))
    except:
        raise HTTPException(status_code=500, detail="Can't get image caption")
