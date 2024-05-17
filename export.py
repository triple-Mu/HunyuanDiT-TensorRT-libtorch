import torch
import torch.nn as nn

from argparse import Namespace
from hydit.inference import End2End
from contextlib import contextmanager


# a useful warp for exporting onnx
@contextmanager
def onnx_export():
    import torch
    import onnx
    from tempfile import TemporaryDirectory
    _export = torch.onnx.export

    def export(
            model,
            args,
            f,
            **kwargs
    ):
        with TemporaryDirectory() as d:
            onnx_file = f'{d}/{f}'
            print(onnx_file)
            _export(model, args, onnx_file, **kwargs)
            onnx_model = onnx.load(onnx_file)
        onnx.save(onnx_model,
                  f,
                  save_as_external_data=True,
                  all_tensors_to_one_file=True,
                  location=f + '.data',
                  convert_attribute=True)

    torch.onnx.export = export
    yield

    torch.onnx.export = _export


@torch.no_grad()
def export():
    # Now only test on such model config
    args = Namespace()
    args.model = 'DiT-g/2'
    args.image_size = [1024, 1024]
    args.infer_mode = 'torch'
    args.load_key = 'ema'
    args.learn_sigma = False
    args.text_states_dim = 1024
    args.text_states_dim_t5 = 2048
    args.text_len = 77
    args.text_len_t5 = 256
    args.norm = 'layer'
    args.sampler = 'ddpm'
    args.noise_schedule = 'scaled_linear'
    args.beta_start = 0.00085
    args.beta_end = 0.03
    args.predict_type = 'v_prediction'
    args.infer_steps = 100
    args.learn_sigma = True
    args.use_fp16 = False
    args.onnx_file = 'hunyuan_unet.onnx'

    # root_path for hunyuan model, download from https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main
    models_root_path = 'models/HunyuanDiT'
    gen = End2End(args, models_root_path)

    # use cpu for avoid OOM
    device = torch.device('cpu')

    latent_model_input = torch.randn(2, 4, 128, 128, device=device)
    t_expand = torch.randint(0, 1000, (2,), dtype=torch.int32, device=device)
    prompt_embeds = torch.randn(2, 77, 1024, device=device)
    attention_mask = torch.ones(2, 77, dtype=torch.bool, device=device)
    prompt_embeds_t5 = torch.randn(2, 256, 2048, device=device)
    attention_mask_t5 = torch.ones(2, 256, dtype=torch.bool, device=device)

    gen.pipeline.unet.float()
    gen.pipeline.unet.to(device)

    class UNET(nn.Module):
        def __init__(self):
            super().__init__()
            self.unet = gen.pipeline.unet

        def forward(self, latent_model_input, t_expand, prompt_embeds, attention_mask,
                    prompt_embeds_t5, attention_mask_t5):
            freqs_cis_img = gen.freqs_cis_img['1024x1024']
            style = torch.zeros(2, dtype=torch.int32, device=device)
            image_meta_size = torch.tensor(
                [[1024, 1024, 1024, 1024, 0, 0],
                 [1024, 1024, 1024, 1024, 0, 0]], dtype=torch.float32, device=device)
            noise_pred = self.unet(
                latent_model_input,
                t_expand,
                encoder_hidden_states=prompt_embeds,
                text_embedding_mask=attention_mask,
                encoder_hidden_states_t5=prompt_embeds_t5,
                text_embedding_mask_t5=attention_mask_t5,
                image_meta_size=image_meta_size,
                style=style,
                cos_cis_img=freqs_cis_img[0],
                sin_cis_img=freqs_cis_img[1],
                return_dict=False,
            )
            return noise_pred

    unet = UNET()
    unet.eval()

    with onnx_export():
        torch.onnx.export(
            unet,
            (latent_model_input, t_expand, prompt_embeds, attention_mask, prompt_embeds_t5, attention_mask_t5),
            args.onnx_file,
            opset_version=17,
            input_names=['latent_model_input', 't_expand', 'prompt_embeds', 'attention_mask', 'prompt_embeds_t5',
                         'attention_mask_t5', 'image_meta_size', 'style', 'cos_cis_img', 'sin_cis_img'],
            output_names=['output'],
        )

    input_ids = torch.randint(0, 100, (2, 77), dtype=torch.int32, device=device)
    attention_mask = torch.ones(2, 77, dtype=torch.bool, device=device)

    args.onnx_file = 'clip_text_encoder.onnx'
    gen.pipeline.text_encoder.float()
    gen.pipeline.text_encoder.to(device)

    gen.pipeline.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)

    torch.onnx.export(
        gen.pipeline.text_encoder,
        (input_ids, attention_mask),
        args.onnx_file,
        opset_version=17,
        input_names=['input_ids', 'attn_mask'],
        output_names=['text_emb'],
    )

    args.onnx_file = 't5_text_encoder.onnx'
    gen.pipeline.embedder_t5.float()
    gen.pipeline.embedder_t5.to(device)

    input_ids = torch.randint(0, 100, (2, 256), dtype=torch.int32, device=device)
    attention_mask = torch.ones(2, 256, dtype=torch.bool, device=device)

    gen.pipeline.embedder_t5(input_ids, attention_mask=attention_mask)

    with onnx_export():
        torch.onnx.export(
            gen.pipeline.embedder_t5,
            (input_ids, attention_mask),
            args.onnx_file,
            opset_version=17,
            input_names=['input_ids', 'attn_mask'],
            output_names=['text_emb_t5'],
        )

    args.onnx_file = 'vae_decoder.onnx'
    gen.pipeline.vae.float()
    gen.pipeline.vae.to(device)

    latent_model_output = torch.randn(1, 4, 128, 128, device=device)
    gen.pipeline.vae.decode(latent_model_output, return_dict=False)

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.vae = gen.pipeline.vae

        def forward(self, latent_model_output):
            return self.vae.decode(latent_model_output, return_dict=False)

    vae = VAE()
    vae.eval()

    torch.onnx.export(
        vae,
        latent_model_output,
        args.onnx_file,
        opset_version=17,
        input_names=['latent_model_output'],
        output_names=['output'],
    )


if __name__ == "__main__":
    export()
