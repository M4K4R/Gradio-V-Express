import os
import cv2
import numpy as np
import torch
import torchaudio.functional
import torchvision.io
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import randn_tensor
from insightface.app import FaceAnalysis
from omegaconf import OmegaConf
from transformers import CLIPVisionModelWithProjection, Wav2Vec2Model, Wav2Vec2Processor

from modules import UNet2DConditionModel, UNet3DConditionModel, VKpsGuider, AudioProjection
from pipelines import VExpressPipeline
from pipelines.utils import draw_kps_image, save_video
from pipelines.utils import retarget_kps


def load_reference_net(unet_config_path, reference_net_path, dtype, device):
    reference_net = UNet2DConditionModel.from_config(unet_config_path).to(dtype=dtype, device=device)
    reference_net.load_state_dict(torch.load(reference_net_path, map_location="cpu"), strict=False)
    print(f'Loaded weights of Reference Net from {reference_net_path}.')
    return reference_net


def load_denoising_unet(unet_config_path, denoising_unet_path, motion_module_path, dtype, device):
    inference_config_path = './inference_v2.yaml'
    inference_config = OmegaConf.load(inference_config_path)
    denoising_unet = UNet3DConditionModel.from_config_2d(
        unet_config_path,
        unet_additional_kwargs=inference_config.unet_additional_kwargs,
    ).to(dtype=dtype, device=device)
    denoising_unet.load_state_dict(torch.load(denoising_unet_path, map_location="cpu"), strict=False)
    print(f'Loaded weights of Denoising U-Net from {denoising_unet_path}.')

    denoising_unet.load_state_dict(torch.load(motion_module_path, map_location="cpu"), strict=False)
    print(f'Loaded weights of Denoising U-Net Motion Module from {motion_module_path}.')

    return denoising_unet


def load_v_kps_guider(v_kps_guider_path, dtype, device):
    v_kps_guider = VKpsGuider(320, block_out_channels=(16, 32, 96, 256)).to(dtype=dtype, device=device)
    v_kps_guider.load_state_dict(torch.load(v_kps_guider_path, map_location="cpu"))
    print(f'Loaded weights of V-Kps Guider from {v_kps_guider_path}.')
    return v_kps_guider


def load_audio_projection(
        audio_projection_path,
        dtype,
        device,
        inp_dim: int,
        mid_dim: int,
        out_dim: int,
        inp_seq_len: int,
        out_seq_len: int,
):
    audio_projection = AudioProjection(
        dim=mid_dim,
        depth=4,
        dim_head=64,
        heads=12,
        num_queries=out_seq_len,
        embedding_dim=inp_dim,
        output_dim=out_dim,
        ff_mult=4,
        max_seq_len=inp_seq_len,
    ).to(dtype=dtype, device=device)
    audio_projection.load_state_dict(torch.load(audio_projection_path, map_location='cpu'))
    print(f'Loaded weights of Audio Projection from {audio_projection_path}.')
    return audio_projection


def get_scheduler():
    inference_config_path = './inference_v2.yaml'
    inference_config = OmegaConf.load(inference_config_path)
    scheduler_kwargs = OmegaConf.to_container(inference_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**scheduler_kwargs)
    return scheduler

class InferenceEngine(object):

    
    def __init__(self, args):
        self.init_params(args)
        self.load_models()
        self.set_generator()
        self.set_vexpress_pipeline()
        self.set_face_analysis_app()

    
    def init_params(self, args):
        for key, value in args.items():
            setattr(self, key, value)

        print("Image width: ", self.image_width)
        print("Image height: ", self.image_height)


    
    def load_models(self):
        self.device = torch.device(f'cuda:{self.gpu_id}')
        self.dtype = torch.float16 if self.dtype == 'fp16' else torch.float32

        self.vae = AutoencoderKL.from_pretrained(self.vae_path).to(dtype=self.dtype, device=self.device)
        print("VAE exists: ", self.vae)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(self.audio_encoder_path).to(dtype=self.dtype, device=self.device)
        self.audio_processor = Wav2Vec2Processor.from_pretrained(self.audio_encoder_path)

        self.scheduler = get_scheduler()
        self.reference_net = load_reference_net(self.unet_config_path, self.reference_net_path, self.dtype, self.device)
        self.denoising_unet = load_denoising_unet(self.unet_config_path, self.denoising_unet_path, self.motion_module_path, self.dtype, self.device)
        self.v_kps_guider = load_v_kps_guider(self.v_kps_guider_path, self.dtype, self.device)
        self.audio_projection = load_audio_projection(
            self.audio_projection_path,
            self.dtype,
            self.device,
            inp_dim=self.denoising_unet.config.cross_attention_dim,
            mid_dim=self.denoising_unet.config.cross_attention_dim,
            out_dim=self.denoising_unet.config.cross_attention_dim,
            inp_seq_len=2 * (2 * self.num_pad_audio_frames + 1),
            out_seq_len=2 * self.num_pad_audio_frames + 1,
        )

        if is_xformers_available():
            self.reference_net.enable_xformers_memory_efficient_attention()
            self.denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

        
    def set_generator(self):
        self.generator = torch.manual_seed(self.seed)

    
    def set_vexpress_pipeline(self):
        print("VAE exists (2): ", self.vae)
        self.pipeline = VExpressPipeline(
            vae=self.vae,
            reference_net=self.reference_net,
            denoising_unet=self.denoising_unet,
            v_kps_guider=self.v_kps_guider,
            audio_processor=self.audio_processor,
            audio_encoder=self.audio_encoder,
            audio_projection=self.audio_projection,
            scheduler=self.scheduler,
        ).to(dtype=self.dtype, device=self.device)

    
    def set_face_analysis_app(self):
        self.app = FaceAnalysis(
            providers=['CUDAExecutionProvider'],
            provider_options=[{'device_id': self.gpu_id}],
            root=self.insightface_model_path,
        )
        self.app.prepare(ctx_id=0, det_size=(self.image_height, self.image_width))

    
    def get_reference_image_for_kps(self, reference_image_path):
        reference_image = Image.open(reference_image_path).convert('RGB')
        print("Image width ???", self.image_width)
        reference_image = reference_image.resize((self.image_height, self.image_width))

        reference_image_for_kps = cv2.imread(reference_image_path)
        reference_image_for_kps = cv2.resize(reference_image_for_kps, (self.image_height, self.image_width))
        reference_kps = self.app.get(reference_image_for_kps)[0].kps[:3]
        return reference_image, reference_image_for_kps, reference_kps
    
    
    def get_waveform_video_length(self, audio_path):
        _, audio_waveform, meta_info = torchvision.io.read_video(audio_path, pts_unit='sec')
        audio_sampling_rate = meta_info['audio_fps']
        print(f'Length of audio is {audio_waveform.shape[1]} with the sampling rate of {audio_sampling_rate}.')
        if audio_sampling_rate != self.standard_audio_sampling_rate:
            audio_waveform = torchaudio.functional.resample(
                audio_waveform,
                orig_freq=audio_sampling_rate,
                new_freq=self.standard_audio_sampling_rate,
            )
        audio_waveform = audio_waveform.mean(dim=0)

        duration = audio_waveform.shape[0] / self.standard_audio_sampling_rate
        video_length = int(duration * self.fps)
        print(f'The corresponding video length is {video_length}.')
        return audio_waveform, video_length
    
    
    def get_kps_sequence(self, kps_path, reference_kps, video_length, retarget_strategy):
        if kps_path != "":
            assert os.path.exists(kps_path), f'{kps_path} does not exist'
            kps_sequence = torch.tensor(torch.load(kps_path))  # [len, 3, 2]
            print(f'The original length of kps sequence is {kps_sequence.shape[0]}.')
            kps_sequence = torch.nn.functional.interpolate(kps_sequence.permute(1, 2, 0), size=video_length, mode='linear')
            kps_sequence = kps_sequence.permute(2, 0, 1)
            print(f'The interpolated length of kps sequence is {kps_sequence.shape[0]}.')
        
        if retarget_strategy == 'fix_face':
            kps_sequence = torch.tensor([reference_kps] * video_length)
        elif retarget_strategy == 'no_retarget':
            kps_sequence = kps_sequence
        elif retarget_strategy == 'offset_retarget':
            kps_sequence = retarget_kps(reference_kps, kps_sequence, only_offset=True)
        elif retarget_strategy == 'naive_retarget':
            kps_sequence = retarget_kps(reference_kps, kps_sequence, only_offset=False)
        else:
            raise ValueError(f'The retarget strategy {retarget_strategy} is not supported.')
        
        return kps_sequence
    
    
    def get_kps_images(self, kps_sequence, reference_image_for_kps, video_length):
        kps_images = []
        for i in range(video_length):
            kps_image = np.zeros_like(reference_image_for_kps)
            kps_image = draw_kps_image(kps_image, kps_sequence[i])
            kps_images.append(Image.fromarray(kps_image))
        return kps_images
    
    def get_video_latents(self, reference_image, kps_images, audio_waveform, video_length, reference_attention_weight, audio_attention_weight):
        vae_scale_factor = 8
        latent_height = self.image_height // vae_scale_factor
        latent_width = self.image_width // vae_scale_factor

        latent_shape = (1, 4, video_length, latent_height, latent_width)
        vae_latents = randn_tensor(latent_shape, generator=self.generator, device=self.device, dtype=self.dtype)

        video_latents = self.pipeline(
            vae_latents=vae_latents,
            reference_image=reference_image,
            kps_images=kps_images,
            audio_waveform=audio_waveform,
            width=self.image_width,
            height=self.image_height,
            video_length=video_length,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            context_frames=self.context_frames,
            context_stride=self.context_stride,
            context_overlap=self.context_overlap,
            reference_attention_weight=reference_attention_weight,
            audio_attention_weight=audio_attention_weight,
            num_pad_audio_frames=self.num_pad_audio_frames,
            generator=self.generator,
        ).video_latents

        return video_latents
    
    
    def get_video_tensor(self, video_latents):
        video_tensor = self.pipeline.decode_latents(video_latents)
        if isinstance(video_tensor, np.ndarray):
            video_tensor = torch.from_numpy(video_tensor)
        return video_tensor
    
    
    def save_video_tensor(self, video_tensor, audio_path, output_path):
        save_video(video_tensor, audio_path, output_path, self.fps)
        print(f'The generated video has been saved at {output_path}.')

    def infer(
            self,
            reference_image_path, audio_path, kps_path,
            output_path,
            retarget_strategy,
            reference_attention_weight, audio_attention_weight):
        reference_image, reference_image_for_kps, reference_kps = self.get_reference_image_for_kps(reference_image_path)
        audio_waveform, video_length = self.get_waveform_video_length(audio_path)
        kps_sequence = self.get_kps_sequence(kps_path, reference_kps, video_length, retarget_strategy)
        kps_images = self.get_kps_images(kps_sequence, reference_image_for_kps, video_length)

        video_latents = self.get_video_latents(
            reference_image, kps_images, audio_waveform,
            video_length,
            reference_attention_weight, audio_attention_weight)
        video_tensor = self.get_video_tensor(video_latents)

        self.save_video_tensor(video_tensor, audio_path, output_path)

