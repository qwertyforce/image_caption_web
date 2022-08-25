import json
from tqdm import tqdm
from os import listdir,getcwd,path,chdir
from PIL import Image
from img_caption_module import image_caption

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('image_path', type=str,nargs='?', default="./../test_images")
args = parser.parse_args()
IMAGE_PATH = args.image_path

def read_img_buffer(image_path):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img
ID_TAGS_ARR=[]
TAG_ONLY_IMPORT=True
file_names = listdir(IMAGE_PATH)
for file_name in tqdm(file_names):
    file_id=int(file_name[:file_name.index('.')])
    img = read_img_buffer(f"{IMAGE_PATH}/{file_name}")
    caption = image_caption(img)
    ID_TAGS_ARR.append({"id":file_id,"caption":caption})

with open('./id_caption.txt', 'w') as outfile:
    json.dump(ID_TAGS_ARR, outfile,ensure_ascii=False)