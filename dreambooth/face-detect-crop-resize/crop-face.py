from PIL import Image
from autocrop import Cropper
import os

DIR_NAME = "ycycy"
image_dir = f"/dreambooth/data/{DIR_NAME}/input/images"
output_dir = f"{image_dir}/cropped"
os.makedirs(output_dir, exist_ok=True)

cropper = Cropper()

total, cropped = 0, 0
failed = []
for filename in os.listdir(image_dir):
    try:
        cropped_image = cropper.crop(os.path.join(image_dir, filename))  # numpy array
        cropped_image = Image.fromarray(cropped_image)
        cropped_image.save(os.path.join(output_dir, filename))
        cropped += 1
    except:
        failed.append(filename)

    total += 1

print(f"Cropped: {cropped} / {total}, Failed: {[f for f in failed]}")
