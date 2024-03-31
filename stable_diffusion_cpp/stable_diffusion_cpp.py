import sd_cpp_lib
from sd_cpp_lib import sd_image_t, sd_type_t, rng_type_t, schedule_t, sample_method_t


class StableDiffusion:
    def __init__(
        self,
        model_path,
        vae_path="",
        taesd_path="",
        control_net_path="",
        lora_model_dir="",
        embed_dir="",
        stacked_id_embed_dir="",
        vae_decode_only=True,
        vae_tiling=False,
        free_params_immediately=False,
        n_threads=-1,
        wtype=sd_type_t.SD_TYPE_F16,
        rng_type=rng_type_t.CUDA_RNG,
        s=schedule_t.DEFAULT,
        keep_clip_on_cpu=False,
        keep_control_net_cpu=False,
        keep_vae_on_cpu=False,
        esrgan_path="",
    ) -> None:
        self.sd_ctx = sdcpy.new_sd_ctx(
            model_path,
            vae_path,
            taesd_path,
            control_net_path,
            lora_model_dir,
            embed_dir,
            stacked_id_embed_dir,
            vae_decode_only,
            vae_tiling,
            free_params_immediately,
            n_threads,
            wtype,
            rng_type,
            s,
            keep_clip_on_cpu,
            keep_control_net_cpu,
            keep_vae_on_cpu,
        )
        self.upscaler_ctx = None
        if esrgan_path:
            self.upscaler_ctx = sdcpy.new_upscaler_ctx(esrgan_path, n_threads, wtype)

    def __del__(self):
        sdcpy.free_sd_ctx(self.sd_ctx)

    def txt_to_img(
        self,
        prompt=[],
        negative_prompt=[],
        clip_skip=-1,
        cfg_scale=7.0,
        width=512,
        height=512,
        sample_method=sample_method_t.EULER_A,
        sample_steps=20,
        seed=42,
        batch_count=1,
        control_cond=None,
        control_strength=0.9,
        style_strength=20.0,
        normalize_input=False,
        input_id_images_path="",
    ):
        prompt_str = ",".join(prompt)
        negative_prompt_str = ",".join(negative_prompt)
        return sdcpy.txt2img(
            self.sd_ctx,
            prompt_str,
            negative_prompt_str,
            clip_skip,
            cfg_scale,
            width,
            height,
            sample_method,
            sample_steps,
            seed,
            batch_count,
            control_cond,
            control_strength,
            style_strength,
            normalize_input,
            input_id_images_path,
        )

    def upscale_img(self, img, upscale_factor):
        if not self.upscaler_ctx:
            return img
        return sdcpy.upscale(self.upscaler_ctx, img, upscale_factor)
