from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load two grayscale images
image1 = Image.open("C:/Users/HP/OneDrive/Desktop/first.png").convert("L")
image2 = Image.open("C:/Users/HP/OneDrive/Desktop/second.png").convert("L")

# Convert images to NumPy arrays
array1 = np.array(image1)
array2 = np.array(image2)

# Perform XOR operation
result_array = np.bitwise_xor(array1, array2)

# Convert the result array back to an image
result_image = Image.fromarray(result_array)

# Display the original and XOR images using matplotlib

#plt.subplot(1, 3, 3)
#plt.imshow(result_array, cmap='gray')
#plt.title('XOR Result')

result_image.show()
