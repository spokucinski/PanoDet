import albumentations as A
import cv2
import os
import shutil

from pathlib import Path

def export_dataset(dataset, name, export_path, use_aug):
    
    print(f"Processing {name} dataset. Count of elements: {len(dataset)}")
    
    for sample in dataset:
        try:
            image_path = sample.filepath
            export_folder = export_path + f"/{name}" 
            if not os.path.exists(export_folder):
                os.makedirs(export_folder)
            shutil.copy(image_path, export_folder)

            if use_aug:
                number_of_augmentations = custom_augmentations[label]
                for i in range(number_of_augmentations):
                    augmented_image = augment_image(image_path)
                    image_name = Path(image_path).stem
                    target_file_path = f"{export_folder}\\{image_name}_aug_{i}.jpg"
                    cv2.imwrite(target_file_path, augmented_image)
        except:
            continue

def augment_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        # A.Affine(mode=cv2.BORDER_WRAP, translate_percent={'x':(0, 1),'y':0}, p=1),
        # A.CoarseDropout(max_holes=5, min_holes=3, max_height=0.1, max_width=0.1, p=0.15),
        # A.HorizontalFlip(p=0.15),
        # A.PixelDropout(dropout_prob=0.01, p=0.15),
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

    transformed =  transform(image=image)
    result = transformed["image"]
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    return result