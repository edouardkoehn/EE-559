import json
import os
import time
from src.dataset import CustomDataset
from src.models import Flamingo0S
from src.utils import ROOT_DIR
from PIL import Image
import requests
from open_flamingo import create_model_and_transforms
import torch
print("####EVAL_1####")
print(time.time())
#Initialize the model
LANG_MODEL_PATH=os.path.join(ROOT_DIR, 'data','pretrained_models','RedPajama-INCITE-Base-3B-v1')
CACHE_MODEL= os.path.join(ROOT_DIR, 'data','pretrained_models')
FLAMINGO_MODEL_PATH=os.path.join(ROOT_DIR, 'data','pretrained_models','OpenFlamingo-3B-vitl-mpt1b', 'checkpoint.pt' )

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path=LANG_MODEL_PATH,
    tokenizer_path=LANG_MODEL_PATH,
    cross_attn_every_n_layers=2,
    cache_dir=os.path.join(ROOT_DIR, 'data','pretrained_models')
)
model.load_state_dict(torch.load(FLAMINGO_MODEL_PATH), strict=False)
demo_image_one = Image.open(
    requests.get(
        "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
    ).raw
)

demo_image_two = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
        stream=True
    ).raw
)

query_image = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028352.jpg", 
        stream=True
    ).raw
)


"""
Step 2: Preprocessing images
Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
 batch_size x num_media x num_frames x channels x height x width. 
 In this case batch_size = 1, num_media = 3, num_frames = 1,
 channels = 3, height = 224, width = 224.
"""
vision_x = [image_processor(demo_image_one).unsqueeze(0), image_processor(demo_image_two).unsqueeze(0), image_processor(query_image).unsqueeze(0)]
vision_x = torch.cat(vision_x, dim=0)
vision_x = vision_x.unsqueeze(1).unsqueeze(0)

"""
Step 3: Preprocessing text
Details: In the text we expect an <image> special token to indicate where an image is.
 We also expect an <|endofchunk|> special token to indicate the end of the text 
 portion associated with an image.
"""
tokenizer.padding_side = "left" # For generation padding tokens should be on the left
lang_x = tokenizer(
    ["<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>An image of"],
    return_tensors="pt",
)


"""
Step 4: Generate text
"""
generated_text = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=20,
    num_beams=3,
)

print("Generated text: ", tokenizer.decode(generated_text[0]))
print(time.time())
"""
print(time.time())
print(os. getcwd())
dataset = CustomDataset(
    csv_file=os.path.join(ROOT_DIR, "data", "MMHS150K", "MMHS150K_text_in_image.csv"),
    img_dir=os.path.join(ROOT_DIR, "data", "MMHS150K", "img_resized/"),
    split="test",
)

model = Flamingo0S(
    config_path=os.path.join(ROOT_DIR, "data", "config", "config_Flamingo0S.json")
)
print(model)

res_dict = {}

for i in range(10):
    prediction = model(dataset[i])
    res_dict[f"{i}"] = prediction

res_path = os.path.join(ROOT_DIR, "data_test.json")
with open(res_path, "w") as f:
    json.dump(res_dict, f)


"""