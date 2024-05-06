import json
import os
import time
from src.dataset import CustomDataset
from src.models import Flamingo0S
from src.utils import ROOT_DIR

from open_flamingo import create_model_and_transforms
import torch

print("####EVAL####_Base model")
print(time.time())

#Initialize the model
LANG_MODEL_PATH=os.path.join(ROOT_DIR, 'data','pretrained_models','RedPajama-INCITE-Base-3B-v1')
CACHE_MODEL= os.path.join(ROOT_DIR, 'data','pretrained_models')
FLAMINGO_MODEL_PATH=os.path.join(ROOT_DIR, 'data','pretrained_models','OpenFlamingo-4B-vitl-rpj3b', 'checkpoint.pt' )

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path=LANG_MODEL_PATH,
    tokenizer_path=LANG_MODEL_PATH,
    cross_attn_every_n_layers=2,
    cache_dir=os.path.join(ROOT_DIR, 'data','pretrained_models')
)
model.load_state_dict(torch.load(FLAMINGO_MODEL_PATH), strict=False)

#Initialize the dataset
dataset = CustomDataset(
    csv_file=os.path.join(ROOT_DIR, "data", "MMHS150K", "MMHS150K_text_in_image.csv"),
    img_dir=os.path.join(ROOT_DIR, "data", "MMHS150K", "img_resized/"),
    split="test"
)


#demo_image_one = dataset[17475]['image']
#demo_image_two = dataset[50550]['image']
#demo_image_three=dataset[37690]['image']
#query_image_one = dataset[4307]['image']
#query_image_two=dataset[36505]['image']

demo_image_one=dataset[4307]['image']
demo_image_two=dataset[50550]['image']
demo_image_three=dataset[37690]['image']
demo_image_four=dataset[21]['image']

query_image_one=dataset[7]['image']
query_image_two=dataset[900]['image']
"""
print('#Test with few shot and meme lable on query image 1')
vision_x = [image_processor(demo_image_one).unsqueeze(0),image_processor(demo_image_one).unsqueeze(0),image_processor(query_image_one).unsqueeze(0)]
vision_x = torch.cat(vision_x, dim=0).unsqueeze(1).unsqueeze(0)
tokenizer.padding_side = "left" # For generation padding tokens should be on the left
lang_x = tokenizer(
    ["<image>Label:Hateful.Explanation: This meme is hateful due to the fact that it targets a specific community<|endofchunk|><image>Label:Not Hateful.Explanation:This meme is humoristic and is not part of hateful speech<|endofchunk|><image>Label:"],
    return_tensors="pt",
)
generated_text = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=25,
    num_beams=3,
)
print("Generated text: ", tokenizer.decode(generated_text[0]))


print('#Test with few shot and meme lable on query image 2')
vision_x = [image_processor(demo_image_one).unsqueeze(0),image_processor(demo_image_one).unsqueeze(0),image_processor(query_image_two).unsqueeze(0)]
vision_x = torch.cat(vision_x, dim=0).unsqueeze(1).unsqueeze(0)
lang_x = tokenizer(
    ["<image>Label:Hateful.Explanation: This meme is hateful due to the fact that it targets a specific community<|endofchunk|><image>Label:Not Hateful.Explanation:This meme is humoristic and is not part of hateful speech<|endofchunk|><image>Label:"],
    return_tensors="pt",
)
generated_text = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=25,
    num_beams=3,
)
print("Generated text: ", tokenizer.decode(generated_text[0]))


print('#Test to explain the image on query1')
vision_x = [image_processor(demo_image_one).unsqueeze(0),image_processor(demo_image_one).unsqueeze(0),image_processor(query_image_one).unsqueeze(0)]
vision_x = torch.cat(vision_x, dim=0).unsqueeze(1).unsqueeze(0)
lang_x = tokenizer(
    ["<image>The text on the image says:Illegal immigration cost the american taxpayer $113 billion annually. A wall on our southern border would pay for itself in a few month<|endofchunk|><image>The text on the image says:You are literally too stupid to insult<|endofchunk|><image>The text on the image says:"],
    return_tensors="pt",
)
generated_text = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=25,
    num_beams=3,
)
print("Generated text: ", tokenizer.decode(generated_text[0]))


print('#Test to explain the image on query2')
vision_x = [image_processor(demo_image_one).unsqueeze(0),image_processor(demo_image_one).unsqueeze(0),image_processor(query_image_two).unsqueeze(0)]
vision_x = torch.cat(vision_x, dim=0).unsqueeze(1).unsqueeze(0)
lang_x = tokenizer(
    ["<image>The text on the image says:Illegal immigration cost the american taxpayer $113 billion annually. A wall on our southern border would pay for itself in a few month<|endofchunk|><image>The text on the image says:You are literally too stupid to insult<|endofchunk|><image>The text on the image says:"],
    return_tensors="pt",
)
generated_text = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=25,
    num_beams=3,
)
print("Generated text: ", tokenizer.decode(generated_text[0]))

print('#Test1  with detailed prompt and one query')
vision_x = [image_processor(query_image_two).unsqueeze(0)]
vision_x = torch.cat(vision_x, dim=0).unsqueeze(1).unsqueeze(0)
lang_x = tokenizer(
    ['''Hatefulness definition: Hate speech is strictly defined as a direct or indirect attack on people based on characteristics, including ethnicity, race, nationality, immigration status, religion, caste, sex, gender identity, sexual orientation, disability, or disease. The definition of attack includes violent or dehumanizing speech, statements of inferiority, and calls for exclusion or segregation. Mocking hate crime is also considered hate speech./nClassification criteria: The meme is hateful if one or both of the visual and textual meanings are hateful. If neither visual nor textual meanings are hateful but together contain implicitly hateful content after combining all meanings, the meme is also hateful. Otherwise, the meme is non-hateful./nFirstly, you explain the meaning of the visual part. Secondly, you explain the meaning of the textual part. Finally, you consider visual and textual meanings to classify the meme as hateful or non-hateful based on the hatefulness definition and classification criteria. Give the probability (from 0 to 1 ) of the meme being hateful.<|endofchunk|><image>'''],
    return_tensors="pt",
)
generated_text = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=30,
    num_beams=3,
)
print("Generated text: ", tokenizer.decode(generated_text[0]))
"""
print('#Test1  with detailed prompt and one query')
vision_x = [image_processor(demo_image_one).unsqueeze(0),
            image_processor(demo_image_two).unsqueeze(0),
            image_processor(demo_image_three).unsqueeze(0),
            image_processor(demo_image_four).unsqueeze(0),
            image_processor(query_image_one).unsqueeze(0)]
vision_x = torch.cat(vision_x, dim=0).unsqueeze(1).unsqueeze(0)
lang_x = tokenizer(
    ['''You are asked to classify a meme as hateful or not hateful. To do it you will:
     1) Describe the visual part
     2) Describe the textual part
     3) Combine the extracted informations from the visual and textual part in order to classify the meme as hateful or not hatefull. 
     Here are some exemples:<|endofchunk|>
     <image>1) We see a white man in a suits. The man is pointing to his head with his finger. 
     2)It' written "Face it replubicans... Your leader is mentally ill"
     3) Hateful<|endofchunk|>
    <image>1): We see a man wearing glasses. The man is in a dialog with an other person. 
     2) It's written "Your are literaly too stupid to insult"
     3) not Hateful<|endofchunk|>
     <image>1): We see a black woman holding a paper. The woman is standing in a large room. 
     2) It's written "on god we gon get you some pussy bro"
     3) not Hateful<|endofchunk|>
     <image>1): We see a white woman sitting in a chair. 
     2) No text on this image
     3) not Hateful<|endofchunk|>
     <image>1)'''],
    return_tensors="pt",
)
generated_text = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=150,
    num_beams=5,
)
print("Generated text: ", tokenizer.decode(generated_text[0]))

print('#Test2  with detailed prompt and one query')
vision_x = [image_processor(demo_image_one).unsqueeze(0),
            image_processor(demo_image_two).unsqueeze(0),
            image_processor(demo_image_three).unsqueeze(0),
            image_processor(demo_image_four).unsqueeze(0),
            image_processor(query_image_two).unsqueeze(0)]
vision_x = torch.cat(vision_x, dim=0).unsqueeze(1).unsqueeze(0)
lang_x = tokenizer(
    ['''You are asked to classify a meme as hateful or not hateful. To do it you will:
     1) Describe the visual part
     2) Describe the textual part
     3) Combine the extracted informations from the visual and textual part in order to classify the meme as hateful or not hatefull. 
     Here are some exemples:<|endofchunk|>
     <image>1) We see a white man in a suits. The man is pointing to his head with his finger. 
     2)It' written "Face it replubicans... Your leader is mentally ill"
     3) Hateful<|endofchunk|>
    <image>1): We see a man wearing glasses. The man is in a dialog with an other person. 
     2) It's written "Your are literaly too stupid to insult"
     3) not Hateful<|endofchunk|>
     <image>1): We see a black woman holding a paper. The woman is standing in a large room. 
     2) It's written "on god we gon get you some pussy bro"
     3) not Hateful<|endofchunk|>
     <image>1): We see a white woman sitting in a chair. 
     2) No text on this image
     3) not Hateful<|endofchunk|>
     <image>1)'''],
    return_tensors="pt",
)

generated_text = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=150,
    num_beams=5,
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