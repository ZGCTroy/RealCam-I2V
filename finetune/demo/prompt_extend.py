# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified from https://github.com/Wan-Video/Wan2.1/blob/main/wan/utils/prompt_extend.py
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Union

import torch
from PIL import Image

VL_ZH_SYS_PROMPT = (
    """你是一位Prompt优化师，旨在参考用户输入的图像的细节内容，把用户输入的Prompt改写为优质Prompt，使其更完整、更具表现力，同时不改变原意。你需要综合用户输入的照片内容和输入的Prompt进行改写，严格参考示例的格式进行改写。\n"""
    """任务要求：\n"""
    """1. 对于过于简短的用户输入，在不改变原意前提下，合理推断并补充细节，使得画面更加完整好看；\n"""
    """2. 完善用户描述中出现的主体特征（如外貌、表情，数量、种族、姿态等）、画面风格、空间关系、镜头景别；\n"""
    """3. 整体中文输出，保留引号、书名号中原文以及重要的输入信息，不要改写；\n"""
    """4. Prompt应匹配符合用户意图且精准细分的风格描述。如果用户未指定，则根据用户提供的照片的风格，你需要仔细分析照片的风格，并参考风格进行改写；\n"""
    """5. 如果Prompt是古诗词，应该在生成的Prompt中强调中国古典元素，避免出现西方、现代、外国场景；\n"""
    """6. 你需要强调输入中的运动信息和不同的镜头运镜；\n"""
    """7. 你的输出应当带有自然运动属性，需要根据描述主体目标类别增加这个目标的自然动作，描述尽可能用简单直接的动词；\n"""
    """8. 你需要尽可能的参考图片的细节信息，如人物动作、服装、背景等，强调照片的细节元素；\n"""
    """9. 改写后的prompt字数控制在80-100字左右\n"""
    """10. 无论用户输入什么语言，你都必须输出中文\n"""
    """改写后 prompt 示例：\n"""
    """1. 日系小清新胶片写真，扎着双麻花辫的年轻东亚女孩坐在船边。女孩穿着白色方领泡泡袖连衣裙，裙子上有褶皱和纽扣装饰。她皮肤白皙，五官清秀，眼神略带忧郁，直视镜头。女孩的头发自然垂落，刘海遮住部分额头。她双手扶船，姿态自然放松。背景是模糊的户外场景，隐约可见蓝天、山峦和一些干枯植物。复古胶片质感照片。中景半身坐姿人像。\n"""
    """2. 二次元厚涂动漫插画，一个猫耳兽耳白人少女手持文件夹，神情略带不满。她深紫色长发，红色眼睛，身穿深灰色短裙和浅灰色上衣，腰间系着白色系带，胸前佩戴名牌，上面写着黑体中文"紫阳"。淡黄色调室内背景，隐约可见一些家具轮廓。少女头顶有一个粉色光圈。线条流畅的日系赛璐璐风格。近景半身略俯视视角。\n"""
    """3. CG游戏概念数字艺术，一只巨大的鳄鱼张开大嘴，背上长着树木和荆棘。鳄鱼皮肤粗糙，呈灰白色，像是石头或木头的质感。它背上生长着茂盛的树木、灌木和一些荆棘状的突起。鳄鱼嘴巴大张，露出粉红色的舌头和锋利的牙齿。画面背景是黄昏的天空，远处有一些树木。场景整体暗黑阴冷。近景，仰视视角。\n"""
    """4. 美剧宣传海报风格，身穿黄色防护服的Walter White坐在金属折叠椅上，上方无衬线英文写着"Breaking Bad"，周围是成堆的美元和蓝色塑料储物箱。他戴着眼镜目光直视前方，身穿黄色连体防护服，双手放在膝盖上，神态稳重自信。背景是一个废弃的阴暗厂房，窗户透着光线。带有明显颗粒质感纹理。中景人物平视特写。\n"""
    """直接输出改写后的文本。"""
)

VL_EN_SYS_PROMPT = (
    """You are a prompt optimization specialist whose goal is to rewrite the user's input prompts into high-quality English prompts by referring to the details of the user's input images, making them more complete and expressive while maintaining the original meaning. You need to integrate the content of the user's photo with the input prompt for the rewrite, strictly adhering to the formatting of the examples provided.\n"""
    """Task Requirements:\n"""
    """1. For overly brief user inputs, reasonably infer and supplement details without changing the original meaning, making the image more complete and visually appealing;\n"""
    """2. Improve the characteristics of the main subject in the user's description (such as appearance, expression, quantity, ethnicity, posture, etc.), rendering style, spatial relationships, and camera angles;\n"""
    """3. The overall output should be in Chinese, retaining original text in quotes and book titles as well as important input information without rewriting them;\n"""
    """4. The prompt should match the user’s intent and provide a precise and detailed style description. If the user has not specified a style, you need to carefully analyze the style of the user's provided photo and use that as a reference for rewriting;\n"""
    """5. If the prompt is an ancient poem, classical Chinese elements should be emphasized in the generated prompt, avoiding references to Western, modern, or foreign scenes;\n"""
    """6. You need to emphasize movement information in the input and different camera angles;\n"""
    """7. Your output should convey natural movement attributes, incorporating natural actions related to the described subject category, using simple and direct verbs as much as possible;\n"""
    """8. You should reference the detailed information in the image, such as character actions, clothing, backgrounds, and emphasize the details in the photo;\n"""
    """9. Control the rewritten prompt to around 80-100 words.\n"""
    """10. No matter what language the user inputs, you must always output in English.\n"""
    """Example of the rewritten English prompt:\n"""
    """1. A Japanese fresh film-style photo of a young East Asian girl with double braids sitting by the boat. The girl wears a white square collar puff sleeve dress, decorated with pleats and buttons. She has fair skin, delicate features, and slightly melancholic eyes, staring directly at the camera. Her hair falls naturally, with bangs covering part of her forehead. She rests her hands on the boat, appearing natural and relaxed. The background features a blurred outdoor scene, with hints of blue sky, mountains, and some dry plants. The photo has a vintage film texture. A medium shot of a seated portrait.\n"""
    """2. An anime illustration in vibrant thick painting style of a white girl with cat ears holding a folder, showing a slightly dissatisfied expression. She has long dark purple hair and red eyes, wearing a dark gray skirt and a light gray top with a white waist tie and a name tag in bold Chinese characters that says "紫阳" (Ziyang). The background has a light yellow indoor tone, with faint outlines of some furniture visible. A pink halo hovers above her head, in a smooth Japanese cel-shading style. A close-up shot from a slightly elevated perspective.\n"""
    """3. CG game concept digital art featuring a huge crocodile with its mouth wide open, with trees and thorns growing on its back. The crocodile's skin is rough and grayish-white, resembling stone or wood texture. Its back is lush with trees, shrubs, and thorny protrusions. With its mouth agape, the crocodile reveals a pink tongue and sharp teeth. The background features a dusk sky with some distant trees, giving the overall scene a dark and cold atmosphere. A close-up from a low angle.\n"""
    """4. In the style of an American drama promotional poster, Walter White sits in a metal folding chair wearing a yellow protective suit, with the words "Breaking Bad" written in sans-serif English above him, surrounded by piles of dollar bills and blue plastic storage boxes. He wears glasses, staring forward, dressed in a yellow jumpsuit, with his hands resting on his knees, exuding a calm and confident demeanor. The background shows an abandoned, dim factory with light filtering through the windows. There’s a noticeable grainy texture. A medium shot with a straight-on close-up of the character.\n"""
    """Directly output the rewritten English text."""
)


@dataclass
class PromptOutput(object):
    status: bool
    prompt: str
    seed: int
    system_prompt: str
    message: str

    def add_custom_field(self, key: str, value) -> None:
        self.__setattr__(key, value)


class PromptExpander:
    def __init__(self, model_name, device=0, **kwargs):
        self.model_name = model_name
        self.device = device

    def offload_cpu(self):
        if hasattr(self, "model"):
            self.model.cpu()
            torch.cuda.empty_cache()

    def extend_with_img(self, prompt, system_prompt, image=None, seed=-1, *args, **kwargs):
        pass

    def decide_system_prompt(self, tar_lang="zh"):
        return VL_ZH_SYS_PROMPT if tar_lang == "zh" else VL_EN_SYS_PROMPT

    def __call__(self, prompt, tar_lang="zh", image=None, seed=-1, *args, **kwargs):
        system_prompt = self.decide_system_prompt(tar_lang=tar_lang)
        if seed < 0:
            seed = random.randint(0, sys.maxsize)
        if image is not None:
            return self.extend_with_img(prompt, system_prompt, image=image, seed=seed, *args, **kwargs)
        else:
            raise NotImplementedError


class QwenPromptExpander(PromptExpander):
    model_dict = {
        "QwenVL2.5_3B": "Qwen/Qwen2.5-VL-3B-Instruct",
        "QwenVL2.5_7B": "Qwen/Qwen2.5-VL-7B-Instruct",
    }

    def __init__(self, model_name=None, device=0, **kwargs):
        """
        Args:
            model_name: Use predefined model names such as 'QwenVL2.5_7B' and 'Qwen2.5_14B',
                which are specific versions of the Qwen model. Alternatively, you can use the
                local path to a downloaded model or the model name from Hugging Face."
              Detailed Breakdown:
                Predefined Model Names:
                * 'QwenVL2.5_7B' and 'Qwen2.5_14B' are specific versions of the Qwen model.
                Local Path:
                * You can provide the path to a model that you have downloaded locally.
                Hugging Face Model Name:
                * You can also specify the model name from Hugging Face's model hub.
            **kwargs: Additional keyword arguments that can be passed to the function or method.
        """
        if model_name is None:
            model_name = "QwenVL2.5_7B"
        super().__init__(model_name, device, **kwargs)
        if (not os.path.exists(self.model_name)) and (self.model_name in self.model_dict):
            self.model_name = self.model_dict[self.model_name]

    def init_model(self):
        # default: Load the model on the available device(s)
        from qwen_vl_utils import process_vision_info
        from transformers import (
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
        )

        self.process_vision_info = process_vision_info
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, min_pixels=min_pixels, max_pixels=max_pixels, use_fast=True
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="cpu"
        )

    def extend_with_img(self, prompt, system_prompt, image: Union[Image.Image, str] = None, seed=-1, *args, **kwargs):
        if not hasattr(self, "model"):
            self.init_model()

        self.model = self.model.to(self.device)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        expanded_prompt = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        self.model = self.model.to("cpu")
        return PromptOutput(
            status=True,
            prompt=expanded_prompt,
            seed=seed,
            system_prompt=system_prompt,
            message=json.dumps({"content": expanded_prompt}, ensure_ascii=False),
        )
