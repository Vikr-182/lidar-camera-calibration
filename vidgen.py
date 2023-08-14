import cv2
import os
from tqdm import tqdm  # Import tqdm

# Path to the directory containing the images
image_folder = 'sample_cam_imgs/'

# Output video filename
output_video = 'trainval.mp4'

# Frame rate (frames per second)
frame_rate = 3

# Get the list of image filenames
image_filenames = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])

# Read the first image to get dimensions
first_image = cv2.imread(os.path.join(image_folder, image_filenames[0]))
height, width, layers = first_image.shape

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 video
out = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

# Write images to video with tqdm progress tracking
for image_filename in tqdm(image_filenames, desc="Writing video"):
    image_path = os.path.join(image_folder, image_filename)
    frame = cv2.imread(image_path)
    out.write(frame)

# Release VideoWriter
out.release()

print("Video created successfully.")
