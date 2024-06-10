import gradio as gr
import shutil
import subprocess

from inference import InferenceEngine
from sequence_utils import extract_kps_sequence_from_video

output_dir = "output"
temp_audio_path = "temp.mp3"


DEFAULT_MODEL_ARGS = {
    'unet_config_path': './model_ckpts/stable-diffusion-v1-5/unet/config.json',
    'vae_path': './model_ckpts/sd-vae-ft-mse/',
    'audio_encoder_path': './model_ckpts/wav2vec2-base-960h/',
    'insightface_model_path': './model_ckpts/insightface_models/',
    'denoising_unet_path': './model_ckpts/v-express/denoising_unet.pth',
    'reference_net_path': './model_ckpts/v-express/reference_net.pth',
    'v_kps_guider_path': './model_ckpts/v-express/v_kps_guider.pth',
    'audio_projection_path': './model_ckpts/v-express/audio_projection.pth',
    'motion_module_path': './model_ckpts/v-express/motion_module.pth',
    #'retarget_strategy': 'fix_face',  # fix_face, no_retarget, offset_retarget, naive_retarget
    'device': 'cuda',
    'gpu_id': 0,
    'dtype': 'fp16',
    'num_pad_audio_frames': 2,
    'standard_audio_sampling_rate': 16000,
    #'reference_image_path': './test_samples/emo/talk_emotion/ref.jpg',
    #'audio_path': './test_samples/emo/talk_emotion/aud.mp3',
    #'kps_path': './test_samples/emo/talk_emotion/kps.pth',
    #'output_path': './output/emo/talk_emotion.mp4',
    'image_width': 512,
    'image_height': 512,
    'fps': 30.0,
    'seed': 42,
    'num_inference_steps': 25,
    'guidance_scale': 3.5,
    'context_frames': 12,
    'context_stride': 1,
    'context_overlap': 4,
    #'reference_attention_weight': 0.95,
    #'audio_attention_weight': 3.0
}

INFERENCE_ENGINE = InferenceEngine(DEFAULT_MODEL_ARGS)

def infer(reference_image, audio_path, kps_sequence_save_path,
        output_path,
        retarget_strategy,
        reference_attention_weight, audio_attention_weight):
    global INFERENCE_ENGINE 
    INFERENCE_ENGINE.infer(
        reference_image, audio_path, kps_sequence_save_path,
        output_path,
        retarget_strategy,
        reference_attention_weight, audio_attention_weight
    )
    return output_path, kps_sequence_save_path

# Function to run V-Express demo
def run_demo(
        reference_image, audio, video,
        kps_path, output_path, retarget_strategy,
        reference_attention_weight=0.95,
        audio_attention_weight=3.0,
        progress=gr.Progress()):
    # Step 1: Extract Keypoints from Video
    progress((0,100), desc="Starting...")

    kps_sequence_save_path = f"{output_dir}/kps.pth"

    if video is not None:
        # Run the script to extract keypoints and audio from the video
        progress((25,100), desc="Extract keypoints and audio...")
        audio_path = video.replace(".mp4", ".mp3")

        extract_kps_sequence_from_video(
            INFERENCE_ENGINE.app,
            video,
            audio_path,
            kps_sequence_save_path
        )

        progress((50,100), desc="Keypoints and audio extracted successfully.")
        #return "Keypoints and audio extracted successfully."
        rem_progress = (75,100)
    else:
        rem_progress = (50,100)
        audio_path = audio
        shutil.copy(kps_path.name, kps_sequence_save_path)

    subprocess.run(["ffmpeg", "-i", audio_path, "-c:v", "libx264", "-crf", "18", "-preset", "slow", temp_audio_path])
    shutil.move(temp_audio_path, audio_path)
    
    # Step 2: Run Inference with Reference Image and Audio
    # Determine the inference script and parameters based on the selected retargeting strategy
    progress(rem_progress, desc="Inference...")

    output_path, kps_sequence_save_path = infer(
        reference_image, audio_path, kps_sequence_save_path,
        output_path,
        retarget_strategy,
        reference_attention_weight, audio_attention_weight
    )

    status = f"Video generated successfully. Saved at: {output_path}"
    progress((100,100), desc=status)
    return output_path, kps_sequence_save_path

# Create Gradio interface
inputs = [
    gr.Image(label="Reference Image", type="filepath"),
    gr.Audio(label="Audio", type="filepath"),
    gr.Video(label="Video"),
    gr.File(label="KPS sequences", value=f"test_samples/short_case/10/kps.pth"),
    gr.Textbox(label="Output Path for generated video", value=f"{output_dir}/output_video.mp4"),
    gr.Dropdown(label="Retargeting Strategy", choices=["no_retarget", "fix_face", "offset_retarget", "naive_retarget"], value="no_retarget"),
    gr.Slider(label="Reference Attention Weight", minimum=0.0, maximum=1.0, step=0.01, value=0.95),
    gr.Slider(label="Audio Attention Weight", minimum=1.0, maximum=5.0, step=0.1, value=3.0)
]

output = [
    gr.Video(label="Generated Video"),
    gr.File(label="Generated KPS Sequences File (kps.pth)")
]

# Title and description for the interface
title = "V-Express Gradio Interface"
description = "An interactive interface for generating talking face videos using V-Express."

# Launch Gradio app
demo = gr.Interface(run_demo, inputs, output, title=title, description=description)
demo.queue().launch()