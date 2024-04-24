import os

import torch
from open_flamingo import create_model_and_transforms
from torch import nn

from src.utils import ROOT_DIR, load_config_model


class Flamingo0S(nn.Module):
    """Class for Flamingo with zero-shot learning"""

    def __init__(self, config_path: os.path) -> None:
        super(Flamingo0S, self).__init__()
        config = load_config_model(config_path)
        self.model_name = config["model_name"]

        LANG_MODEL_PATH = os.path.join(
            ROOT_DIR, "data", "pretrained_models", config["language_model"]
        )
        CACHE_MODEL = os.path.join(ROOT_DIR, "data", "pretrained_models")
        FLAMINGO_MODEL_PATH = os.path.join(
            ROOT_DIR,
            "data",
            "pretrained_models",
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

    def forward(self, x, train=False):
        """Method get the inferance from the model
        Args : x(dict): containing the keys:image, labels, tweet_text, img_text
        """
        visual_input = [self.image_processor(x["image"]).unsqueeze(0)]
        visual_input = torch.cat(visual_input, dim=0)
        visual_input = visual_input.unsqueeze(1).unsqueeze(0)

        text_input = [''.join([self.prompt_text, x["tweet_text"]])]
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