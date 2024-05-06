from transformers import pipeline
from PIL import Image    
import requests
from src.dataset import CustomDataset
from src.models import Flamingo0S
from src.utils import ROOT_DIR
import os
import matplotlib.pyplot as plt
print("####### Lava#####")

dataset = CustomDataset(
    csv_file=os.path.join(ROOT_DIR, "data", "MMHS150K", "MMHS150K_text_in_image.csv"),
    img_dir=os.path.join(ROOT_DIR, "data", "MMHS150K", "img_resized/"),
    split="test"
)
model_id = "llava-hf/llava-1.5-7b-hf"

pipe=pipeline(task= "image-to-text", 
                      model= "C:\\Users\\me\\mymodel",
                      tokenizer = "C:\\Users\\me\\mymodel")
pipe = pipeline("image-to-text", model=model_id)
image_1=dataset[6]['image']
text_1=dataset[6]['text']
image_2=dataset[0]['image']
image_3=dataset[1]['image']
image_4=dataset[2]['image']
#test1
prompt = "USER: <image>\nDo you think that this image is a hateful meme? Please format you answer as following :1)Hateful or not Hateful 2) Explaination\nASSISTANT:"
outputs = pipe(image_1, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
print(outputs)
#test2 same prompt
prompt = "USER: <image>\nDo you think that this image is a hateful meme? Please format you answer as following :1)Hateful or not Hateful 2) Explaination\nASSISTANT:"
outputs = pipe(image_2, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
print(outputs)

#test2 same prompt
prompt = "USER: <image>\nDo you think that this image is a hateful meme? Please format you answer as following :1)Hateful or not Hateful 2) Explaination\nASSISTANT:"
outputs = pipe(image_3, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
print(outputs)

#test2 same prompt
prompt = "USER: <image>\nDo you think that this image is a hateful meme? Please format you answer as following :1)Hateful or not Hateful 2) Explaination\nASSISTANT:"
outputs = pipe(image_4, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
print(outputs)

#test3 different prompte
prompt = "USER: <image>\nDo you think that this image is a hateful meme? Please format you answer as following :1)Hateful or not Hateful 2) Explaination\nASSISTANT:"
outputs = pipe(image_4, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
print(outputs)
