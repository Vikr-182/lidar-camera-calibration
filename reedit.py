import os
import cv2
from tqdm import tqdm

for num in tqdm(range(22020)):
    imgfile = '{:06}'.format(num) + '.png'
    img = cv2.imread('sample_cam_imgs/' + imgfile)
    demn = img[450:, 375:945, :].copy()
    img[450:, 375:945, :] = img[450:, 1546:2116, :]
    img[450:, 1546:2116, :] = demn
    cv2.imwrite('sample_cam_imgs_fixed/' + imgfile, img)

image_folder = 'sample_cam_imgs_fixed/'
output_video = 'trainval.mp4'
frame_rate = 3
image_filenames = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
first_image = cv2.imread(os.path.join(image_folder, image_filenames[0]))
height, width, layers = first_image.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 video
out = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

for image_filename in tqdm(image_filenames, desc="Writing video"):
    image_path = os.path.join(image_folder, image_filename)
    frame = cv2.imread(image_path)
    out.write(frame)

out.release()
print("Video created successfully.")
