import os
import shutil
import random

# SOURCE TRAIN PATH (FER2013)
source_path = r"C:\Users\User\Downloads\archive\train"

# DESTINATION PATH (YOUR DATASET)
dest_path = r"C:\Users\User\Documents\EBOAGWU_23CG034057_EMOTION_DETECTION_WEB_APP\emotion_dataset"

emotions = ["angry", "sad", "happy", "fear"]  # 4 emotion choices
num_images = 30  # number to extract per emotion

for emotion in emotions:
    src_folder = os.path.join(source_path, emotion)
    dst_folder = os.path.join(dest_path, emotion)

    images = os.listdir(src_folder)
    selected = random.sample(images, num_images)  # pick 30 random images

    for img in selected:
        shutil.copy(os.path.join(src_folder, img), os.path.join(dst_folder, img))

print("✅ DONE. IMAGES SUCCESSFULLY COPIED.")





