from moviepy import VideoFileClip
import os

"""
Converts manim videos to gifs to be showcased
"""

video_dir = "./media/videos/main/480p15"
gif_output_dir = "./media/gifs"

os.makedirs(gif_output_dir, exist_ok=True)

# Get list of video files
video_files = [
    os.path.join(video_dir, f)
    for f in os.listdir(video_dir)
    if os.path.isfile(os.path.join(video_dir, f))
]

print("Video files found:")
print(video_files)

# Convert each video to GIF
for video_path in video_files:
    try:
        clip = VideoFileClip(video_path)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        gif_path = os.path.join(gif_output_dir, f"{base_name}.gif")
        clip.write_gif(gif_path, fps=11.5)
        print(f"Exported GIF: {gif_path}")
        clip.close()
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
