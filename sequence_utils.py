import os
import cv2
import torch
from imageio_ffmpeg import get_ffmpeg_exe

def extract_kps_sequence_from_video(
        face_analysis_app,
        video_path, audio_save_path,
        kps_sequence_save_path):
    os.system(f'{get_ffmpeg_exe()} -i "{video_path}" -y -vn "{audio_save_path}"')

    kps_sequence = []
    video_capture = cv2.VideoCapture(video_path)
    frame_idx = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        faces = face_analysis_app.get(frame)
        assert len(faces) == 1, f'There are {len(faces)} faces in the {frame_idx}-th frame. Only one face is supported.'

        kps = faces[0].kps[:3]
        kps_sequence.append(kps)
        frame_idx += 1
    torch.save(kps_sequence, kps_sequence_save_path)

