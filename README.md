# Gradio V-Express Project

## Introduction

This repository is created to run the V-Express project via Gradio, compiling various resources and code snippets shared by different individuals. The original project was created by [tencent-ailab](https://github.com/tencent-ailab/V-Express) and the contributions made by [faraday](https://github.com/faraday) and [StableAIHub](https://www.youtube.com/@StableAIHub) made it possible to run this project using Gradio.

Check out the YouTube Video NewGenAI:
[NewGenAI](https://youtu.be/OFt6a2rR8GY?si=S82ZwP1w1OJvlYJR)

## Installation

### Steps

1. Download CMake from [here](https://cmake.org/download/) and add System Variables Path `C:\Program Files\CMake\bin`.

2. Download ffmpeg from [here](https://ffmpeg.org/download.html) and add System Variables Path `C:\Program Files\ffmpeg\bin`.

3. Clone the repository:
   ```bash
   git clone https://github.com/M4K4R/Gradio-V-Express
   ```
4. Navigate inside the cloned repository:
   ```bash
   cd Gradio-V-Express
   ```
5. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
6. Activate the virtual environment:
   ```bash
   .\venv\Scripts\activate
   ```
7. Install the required packages:

   ```bash
   pip install xformers==0.0.21
   pip install torch==2.0.1+cu118 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
   pip install diffusers==0.24.0 imageio-ffmpeg==0.4.9 omegaconf==2.2.3 onnxruntime-gpu==1.16.3 safetensors==0.4.2 transformers==4.30.2 einops==0.4.1 tqdm==4.66.1 av==11.0.0 accelerate
   ```

8. Download the Insightface prebuilt wheel from [here](https://github.com/Gourieff/Assets/tree/main/Insightface).

9. Install Insightface:

   ```bash
   pip install "C:\sd\insightface-0.7.3-cp310-cp310-win_amd64.whl"
   ```

10. Download the Dlib prebuilt wheel from [here](https://github.com/z-mahmud22/Dlib_Windows_Python3.x).

11. Install Dlib:

    ```bash
    pip install "C:\sd\dlib-19.22.99-cp310-cp310-win_amd64.whl"
    ```

12. Download the models:

    ```bash
    git lfs install
    git clone https://huggingface.co/tk93/V-Express
    move V-Express/model_ckpts model_ckpts
    move V-Express/*.bin model_ckpts/v-express
    ```

13. Install Gradio web UI:
    ```bash
    pip install gradio
    ```

## Usage

1. To start the application with Gradio, run:

   ```bash
   python app.py
   ```

2. Open your web browser and navigate to the provided Gradio link to interact with the application.

## License

This repository does not contain original code, but rather a collection of resources from various contributors. Please refer to the individual licenses of the original projects for more details.

## Contributions

If you have any suggestions or questions, feel free to open an issue or submit a pull request. We welcome contributions from the community!
