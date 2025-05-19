import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

SCREEN_ROOT = Path("tests/screen_data")


@contextmanager
def create_screen_data(
    duration=10.0, frame_shape=(32, 32), fps=10.0, image_frame_count=10, num_videos=1
):

    try:
        data_dir = SCREEN_ROOT / "data"
        meta_dir = SCREEN_ROOT / "meta"
        data_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        n_frames = int(np.round(duration * fps))
        timestamps = np.linspace(0.0, duration, n_frames, endpoint=False)
        np.save(SCREEN_ROOT / "timestamps.npy", timestamps)

        frames = np.random.rand(n_frames, *frame_shape).astype(np.float32)

        # Save image frames
        for i in range(min(image_frame_count, n_frames)):
            np.save(data_dir / f"{i:05d}.npy", frames[i])
            with open(meta_dir / f"{i:05d}.yml", "w") as f:
                yaml.dump(
                    {
                        "first_frame_idx": i,
                        "image_size": list(frame_shape),
                        "modality": "image",
                        "tier": "train",
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
                np.save(data_dir / f"{vid_idx+image_frame_count:05d}.npy", video_array)
                with open(meta_dir / f"{vid_idx+image_frame_count:05d}.yml", "w") as f:
                    yaml.dump(
                        {
                            "first_frame_idx": start_vid + start,
                            "num_frames": len(chunk),
                            "image_size": [video_array.shape[1], video_array.shape[2]],
                            "modality": "video",
                            "tier": "train",
                        },
                        f,
                    )

        with open(SCREEN_ROOT / "meta.yml", "w") as f:
            yaml.dump(
                {
                    "modality": "screen",
                    "frame_rate": fps,
                },
                f,
            )

        yield timestamps

    finally:
        shutil.rmtree(SCREEN_ROOT)
