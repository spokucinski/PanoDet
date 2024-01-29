from PIL import Image
import os
import scipy as sc
from scipy import ndimage as nd
from matplotlib import pyplot as plt
import cv2
# import albumentations as A


#os.chdir("Dataset_Exporter")
IMG_PATH = "/workspaces/PanoDet/Dataset_Exporter/data/sample/ApartmentAirB&B/images/Kiok64M7Jn1-6a6a5b76440e4e008e8d63600810f576.jpg"

# image = Image.open(IMG_PATH)
# image.show()

# updated = image.transform(image.size, Image.AFFINE, (0, 0, -50, 0, 1, 0))
# updated.show()

image = cv2.imread(IMG_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#cv2.namedWindow('source', cv2.WINDOW_NORMAL)
cv2.imshow("source", image)

transform = A.Compose([
    A.Affine(mode=cv2.BORDER_WRAP, translate_percent={'x':(0, 1),'y':0}, p=1),
    A.CoarseDropout(max_holes=5, min_holes=3, max_height=0.1, max_width=0.1, p=0.15),
    A.HorizontalFlip(p=0.15),
    A.PixelDropout(dropout_prob=0.01, p=0.15),
    A.Blur(blur_limit=15, p=0.15),
    A.CLAHE(p=0.15),
    A.ColorJitter(brightness=0, contrast=0, saturation=0.1, hue=0.1, p=0.15),
    A.Equalize(mode='cv', by_channels=False, p=0.15),
    A.GaussNoise(var_limit=(50, 250), p=0.2),
    A.Sharpen(alpha=(0.05, 0.1), p=0.15),
    A.ToGray(p=0.15),
    A.RandomBrightness(limit=0.25, p=0.25),
    A.RandomContrast(limit=0.25, p=0.25)
])

for i in range(100):
    transformed =  transform(image=image)
    result = transformed["image"]
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"C:\\Users\\Sebastian\\Documents\\PanoDet\\Dataset_Exporter\\augmentation_tests\\{i}.jpg", result)

transformA = A.Compose([
    A.Affine(mode=cv2.BORDER_WRAP, translate_percent={'x':(0, 1),'y':0})
])

transformB = A.Compose([
    A.Affine(mode=cv2.BORDER_WRAP, translate_percent={'x':(0, 1),'y':0})
])

transformC = A.Compose([
    A.Affine(mode=cv2.BORDER_WRAP, translate_percent={'x':(0, 1),'y':0})
])

transformD = A.Compose([
    A.Affine(mode=cv2.BORDER_WRAP, translate_percent={'x':(0, 1),'y':0})
])

transformedA = transform(image=image)
transformedB = transform(image=image)
transformedC = transform(image=image)
transformedD = transform(image=image)

transformed_imageA = transformedA["image"]
transformed_imageB = transformedB["image"]
transformed_imageC = transformedC["image"]
transformed_imageD = transformedD["image"]

cv2.namedWindow('transformedA', cv2.WINDOW_NORMAL)
cv2.resizeWindow('transformedA', 600, 300)
cv2.imshow("transformedA", transformed_imageA)

cv2.namedWindow('transformedB', cv2.WINDOW_NORMAL)
cv2.resizeWindow('transformedB', 600, 300)
cv2.imshow("transformedB", transformed_imageB)

cv2.namedWindow('transformedC', cv2.WINDOW_NORMAL)
cv2.resizeWindow('transformedC', 600, 300)
cv2.imshow("transformedC", transformed_imageC)

cv2.namedWindow('transformedD', cv2.WINDOW_NORMAL)
cv2.resizeWindow('transformedD', 600, 300)
cv2.imshow("transformedD", transformed_imageD)


cv2.waitKey()

# f, axarr = plt.subplots(2, 1)

# image = plt.imread(IMG_PATH)
# shifted = nd.shift(image, [0, 100, 0], mode='wrap')

# axarr[0].imshow(image)
# axarr[1].imshow(shifted)
# plt.show()

i=3
