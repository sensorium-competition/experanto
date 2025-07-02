import shutil
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

SCREEN_ROOT = Path("tests/screen_data")


@contextmanager
def create_screen_data(
    duration=10.0,
    frame_shape=(32, 32),
    fps=10.0,
    image_frame_count=10,
    num_videos=1,
    encoded=False,
):
    try:
        data_dir = SCREEN_ROOT / "data"
        meta_dir = SCREEN_ROOT / "meta"
        data_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        n_frames = int(np.floor(duration * fps))
        timestamps = np.linspace(0.0, duration, n_frames, endpoint=False)
        np.save(SCREEN_ROOT / "timestamps.npy", timestamps)

        # Generate frames with values in [0, 255] for better encoding
        frames = (np.random.rand(n_frames, *frame_shape) * 255).astype(np.uint8)

        # Save image frames
        for i in range(min(image_frame_count, n_frames)):
            if encoded:
                # Save as JPEG
                img = Image.fromarray(
                    frames[i], mode="L" if len(frame_shape) == 2 else "RGB"
                )
                img.save(data_dir / f"{i:05d}.jpg", "JPEG", quality=95)
                file_ext = ".jpg"
            else:
                # Save as numpy array (original behavior)
                np.save(data_dir / f"{i:05d}.npy", frames[i].astype(np.float32))
                file_ext = ".npy"

            with open(meta_dir / f"{i:05d}.yml", "w") as f:
                yaml.dump(
                    {
                        "first_frame_idx": i,
                        "image_size": list(frame_shape),
                        "modality": "image",
                        "tier": "train",
                        "file_format": file_ext,
                        "encoded": encoded,
                    },
                    f,
                )

        # Save video frames for remainder
        start_vid = image_frame_count
        if start_vid < n_frames:
            video_frames = frames[start_vid:]
            frames_per_vid = int(np.ceil(len(video_frames) / num_videos))

            for vid_idx in range(num_videos):
                start = vid_idx * frames_per_vid
                end = min(start + frames_per_vid, len(video_frames))
                chunk = video_frames[start:end]

                if len(chunk) == 0:
                    continue

                video_array = np.stack(chunk, axis=0)
                file_idx = vid_idx + image_frame_count

                if encoded:
                    # Save as MP4 using OpenCV (torchcodec compatible)
                    mp4_path = data_dir / f"{file_idx:05d}.mp4"
                    _save_mp4_opencv(video_array, mp4_path, fps)
                    file_ext = ".mp4"
                else:
                    # Save as numpy array (original behavior)
                    np.save(
                        data_dir / f"{file_idx:05d}.npy", video_array.astype(np.float32)
                    )
                    file_ext = ".npy"

                with open(meta_dir / f"{file_idx:05d}.yml", "w") as f:
                    yaml.dump(
                        {
                            "first_frame_idx": start_vid + start,
                            "num_frames": len(chunk),
                            "image_size": [video_array.shape[1], video_array.shape[2]],
                            "modality": "video",
                            "tier": "train",
                            "file_format": file_ext,
                            "encoded": encoded,
                            "fps": fps,
                        },
                        f,
                    )

        with open(SCREEN_ROOT / "meta.yml", "w") as f:
            yaml.dump(
                {
                    "modality": "screen",
                    "frame_rate": fps,
                    "encoded": encoded,
                },
                f,
            )

        yield timestamps
    finally:
        shutil.rmtree(SCREEN_ROOT)


def _save_mp4_opencv(video_array, output_path, fps):
    if len(video_array.shape) == 3:
        # Grayscale: add channel dimension and convert to RGB needed for conversion
        video_array = np.expand_dims(video_array, axis=-1)
        video_array = np.repeat(video_array, 3, axis=-1)

    height, width = video_array.shape[1], video_array.shape[2]

    # Use H.264 codec which torchcodec supports
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), isColor=True)

    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    try:
        for frame in video_array:
            # Convert from RGB to BGR for OpenCV
            if frame.shape[-1] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame

            out.write(frame_bgr)
    finally:
        out.release()

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to create valid MP4 file at {output_path}")
