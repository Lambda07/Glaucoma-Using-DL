from PIL import Image
import cv2
import numpy as np
import os

mask = 'C:/Users/HP/OneDrive/Desktop/vinay/training/GT'
image = 'C:/Users/HP/OneDrive/Desktop/vinay/training/NORMAL'
# Load the grayscale image
m = []
p = []
for filename in os.listdir(mask):
    img_path = os.listdir(mask+'/'+filename+'/SoftMap')
    first = Image.open(mask+'/'+filename+'/SoftMap/'+img_path[0])
    second = Image.open(mask+'/'+filename+'/SoftMap/'+img_path[1])
    new_size = (2000, 1710)
    resized_image_1 = first.resize(new_size)
    resized_image_2 = second.resize(new_size)
    array1 = np.array(resized_image_1)
    array2 = np.array(resized_image_2)
    color_image_1 = cv2.merge([np.zeros_like(array1), array1, np.zeros_like(array1)])
    color_image_2 = cv2.merge([array2, np.zeros_like(array2), np.zeros_like(array2)])
    array1=color_image_1
    array2=color_image_2
    result_array = np.bitwise_xor(array1, array2)
    result_image = Image.fromarray(result_array)
    m.append(result_image)

for filename in os.listdir(image):
    img_path = image+'/'+filename
    first = Image.open(img_path)
    new_size = (2000, 1710)
    resized_image_1 = first.resize(new_size)
    p.append(resized_image_1)




destination_directory_pos = 'C:/Users/HP/OneDrive/Desktop/unet_images/image'
destination_directory_neg = 'C:/Users/HP/OneDrive/Desktop/unet_images/mask'

for i in range(len(p)):
    destination_path_pos = os.path.join(destination_directory_pos, "image_"+str(i)+".png")
    p[i].save(destination_path_pos)

for i in range(len(m)):
    destination_path_neg = os.path.join(destination_directory_neg, "image_"+str(i)+".png")
    m[i].save(destination_path_neg)

