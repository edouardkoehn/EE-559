import os

import torch
from open_flamingo import create_model_and_transforms
from torch import nn
from torchvision.transforms.functional import to_pil_image as to_pil
from transformers import LlavaForConditionalGeneration, LlavaProcessor

from src.utils import ROOT_DIR, load_config_model


class Flamingo0S(nn.Module):
    """Class for Flamingo with zero-shot learning"""

    def __init__(self, config_path: os.path) -> None:
        super(Flamingo0S, self).__init__()
        config = load_config_model(config_path)
        self.model_name = config["model_name"]

        LANG_MODEL_PATH = os.path.join(
            ROOT_DIR, "data", "pretrained_model", config["language_model"]
        )
        CACHE_MODEL = os.path.join(ROOT_DIR, "data", "pretrained_model")
        FLAMINGO_MODEL_PATH = os.path.join(
            ROOT_DIR,
            "data",
            "pretrained_model",
            "OpenFlamingo-3B-vitl-mpt1b",
            "checkpoint.pt",
        )

        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path=config["vision_model"],
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=LANG_MODEL_PATH,
            tokenizer_path=LANG_MODEL_PATH,
            cross_attn_every_n_layers=config["n_cros_attention"],
            cache_dir=CACHE_MODEL,
        )
        self.model.load_state_dict(torch.load(FLAMINGO_MODEL_PATH), strict=False)
        self.model.eval()
        self.tokenizer.padding_side = "left"

        self.prompt_text = config["prompting"]

    def initialize_prompt(self, dataset):
        ex_img = []
        ex_text = []
        print(self.prompt_ex)
        for id, prompt in self.prompt_ex.items():
            print(id, prompt)
            idx = dataset.get_index(id)
            ex_img.append(self.image_processor(dataset[idx]["image"]).unsqueeze(0))
            ex_text.append(prompt)
        ex_img = torch.cat(ex_img, dim=0).unsqueeze(1).unsqueeze(0)
        ex_text = "<|endofchunk|>".join(ex_text)
        ex_token = self.tokenizer([ex_text], return_tensors="pt")

        generated_text = self.model.generate(
            vision_x=ex_img,
            lang_x=ex_token["input_ids"],
            attention_mask=ex_token["attention_mask"],
            max_new_tokens=20,
            num_beams=3,
        )
        return self.tokenizer.decode(generated_text[0])

    def forward(self, x, train=False):
        """Method get the inferance from the model
        Args : x(dict): containing the keys:image, labels, tweet_text, img_text
        """
        visual_input = [self.image_processor(x["image"]).unsqueeze(0)]
        visual_input = torch.cat(visual_input, dim=0)
        visual_input = visual_input.unsqueeze(1).unsqueeze(0)

        text_input = ["".join([self.prompt_text])]
        text_input = self.tokenizer(text_input, return_tensors="pt")
        print(text_input)
        generated_text = self.model.generate(
            vision_x=visual_input,
            lang_x=text_input["input_ids"],
            attention_mask=text_input["attention_mask"],
            max_new_tokens=20,
            num_beams=3,
        )
        return {"generation": self.tokenizer.decode(generated_text[0])}


class Lava(nn.Module):
    """Class for Flamingo with zero-shot learning"""

    def __init__(self, config_path: os.path, device) -> None:
        super(Lava, self).__init__()
        self.processor = LlavaProcessor.from_pretrained(
            os.path.join(ROOT_DIR, "data", "pretrained_model", "llava-1.5-7b-hf")
        )
        self.model = LlavaForConditionalGeneration.from_pretrained(
            os.path.join(ROOT_DIR, "data", "pretrained_model", "llava-1.5-7b-hf")
        ).to(device)
        self.model.low_cpu_mem_usage = True
        config = load_config_model(config_path)
        self.use_tweet_text = config["use_tweete_text"]
        self.use_image = config["use_image"]
        self.prompt_text = config["prompting"].replace("'", '"')
        self.device = device

    def forward(self, x, train=False):
        """Method get the inferance from the model
        only accepts batch inputs
        Args : x(dict): containing the keys:image, labels, tweet_text, img_text
        """
        if self.use_image:
            visual_input = [0 for i in range(x["image"].shape[0])]
            for i in range(x["image"].shape[0]):
                visual_input[i] = to_pil(x["image"][i, :, :, :])
        else:
            visual_input = [0 for i in range(x["image"].shape[0])]
            for i in range(x["image"].shape[0]):
                visual_input[i] = to_pil(torch.zeros(x["image"][i, :, :, :].shape))

        if self.use_tweet_text:
            prompt = [
                "".join([self.prompt_text, x["tweet_text"][i], "\nASSISTANT:"])
                for i in range(x["image"].shape[0])
            ]
        else:
            prompt = [self.prompt_text for i in range(x["image"].shape[0])]

        inputs = self.processor(
            prompt, visual_input, return_tensors="pt", padding=True
        ).to(self.device)
        output = self.model.generate(**inputs, do_sample=False)
        generated_text = self.processor.batch_decode(output, skip_special_tokens=True)

        return {"generation": generated_text, "index": x["index"]}
