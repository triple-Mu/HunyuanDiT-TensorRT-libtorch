import py_hunyuan_dit as hunyuan
import torch
from PIL import Image
from transformers import AutoTokenizer

base_path = 'HunyuanDiT'
bert_tokenizer = AutoTokenizer.from_pretrained('HunyuanDiT/t2i/tokenizer')
t5_tokenizer = AutoTokenizer.from_pretrained('HunyuanDiT/t2i/mt5')

string = '一只正在吃竹子的大熊猫'
bert_input_ids = bert_tokenizer(string, padding="max_length", max_length=77).input_ids
t5_input_ids = t5_tokenizer(string, padding="max_length", max_length=256).input_ids

pipeline = hunyuan.Pipeline(
    base_path + '/clip_text_encoder.plan',
    base_path + '/t5_text_encoder.plan',
    base_path + '/hunyuan_unet.plan',
    base_path + '/vae_decoder.plan',
    42
)

img = pipeline.generate(bert_input_ids, t5_input_ids, None, None, 100)
Image.fromarray(img).show()
