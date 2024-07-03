from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil

positive = 'C:/Users/HP/OneDrive/Desktop/temp_train/positive'
negative = 'C:/Users/HP/OneDrive/Desktop/temp_train/negative'

pos = []
neg = []
for filename in os.listdir(positive):
    img_path = os.listdir(positive+'/'+filename+'/SoftMap')
    first = Image.open(positive+'/'+filename+'/SoftMap/'+img_path[0])
    second = Image.open(positive+'/'+filename+'/SoftMap/'+img_path[1])
    new_size = (2000, 1710)
    resized_image_1 = first.resize(new_size)
    resized_image_2 = second.resize(new_size)
    array1 = np.array(resized_image_1)
    array2 = np.array(resized_image_2)
    result_array = np.bitwise_xor(array1, array2)
    result_image = Image.fromarray(result_array)
    pos.append(result_image)
    
# Now, loaded_images is a list containing PIL Image objects of all the images in the specified folder.

for filename in os.listdir(negative):
    img_path = os.listdir(negative+'/'+filename+'/SoftMap')
    first = Image.open(negative+'/'+filename+'/SoftMap/'+img_path[0])
    second = Image.open(negative+'/'+filename+'/SoftMap/'+img_path[1])
    new_size = (2000, 1710)
    resized_image_1 = first.resize(new_size)
    resized_image_2 = second.resize(new_size)
    array1 = np.array(resized_image_1)
    array2 = np.array(resized_image_2)
    result_array = np.bitwise_xor(array1, array2)
    result_image = Image.fromarray(result_array)
    neg.append(result_image)


# Destination directory on your system
destination_directory_pos = 'C:/Users/HP/OneDrive/Desktop/xor_images/positive'
destination_directory_neg = 'C:/Users/HP/OneDrive/Desktop/xor_images/negative'
for i in range(len(pos)):
    destination_path_pos = os.path.join(destination_directory_pos, "image_"+str(i)+".png")
    pos[i].save(destination_path_pos)
