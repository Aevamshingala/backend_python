from moviepy import *
from PIL import Image

# your image list
images = [
    r"D:\editsoft\backend\temp_media\img1.jpg",
    r"D:\editsoft\backend\temp_media\img2.jpg",
   
]

# Resize all images to same size (e.g., 1280x720)
standard_size = (1280, 720)

resized_files = []
for i, path in enumerate(images):
    img = Image.open(path)
    img = img.resize(standard_size)
    new_path = f"resized_{i}.jpg"
    img.save(new_path)
    resized_files.append(new_path)

# Now create clip
video = ImageSequenceClip(resized_files, fps=24)
audio = AudioFileClip(r"D:\editsoft\backend\audio\02.mp3")

final = video.with_audio(audio)
final.write_videofile(r"D:\editsoft\backend\output.mp4", fps=24)
