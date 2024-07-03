from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import cv2

positive = 'C:/Users/HP/OneDrive/Desktop/vinay/Training/NORMAL'

pos = []
neg = []
for filename in os.listdir(positive):
    image1 = Image.open(positive+"/"+filename)
    new_size = (512, 512)
    resized_image_1 = image1.resize(new_size)
    pos.append(resized_image_1)
    
# Now, loaded_images is a list containing PIL Image objects of all the images in the specified folder.


# Destination directory on your system
destination_directory_pos = 'C:/Users/HP/OneDrive/Desktop/colour_mask/images'
for i in range(len(pos)):
    destination_path_pos = os.path.join(destination_directory_pos, "image_"+str(i)+".png")
    pos[i].save(destination_path_pos)
