import cv2
import os
import face_recognition
import random
import numpy as np

def augment_and_save_faces(input_dir, output_dir,
                          flip_probability=0.5,
                          brightness_range=(0.7, 1.3),
                          scale_range=(0.8, 1.2),
                          rotation_range=(-15, 15),
                          contrast_range=(0.8, 1.2),
                          num_augmentations=10): 
    """
    Detects faces, applies random augmentations, and saves results.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(input_dir, filename)
            original_image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_image)

            for i in range(num_augmentations):
                augmented_image = augment_image(
                    original_image.copy(),
                    flip=random.random() < flip_probability,
                    brightness_factor=random.uniform(*brightness_range),
                    scale_factor=random.uniform(*scale_range),
                    rotation_angle=random.randint(*rotation_range),
                    contrast_factor=random.uniform(*contrast_range)
                )
                augmented_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.jpg"
                augmented_path = os.path.join(output_dir, augmented_filename)
                cv2.imwrite(augmented_path, augmented_image)

                if len(face_locations) > 0:
                    # Crop and save faces from augmented images
                    rgb_augmented = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
                    face_locations_augmented = face_recognition.face_locations(rgb_augmented)
                    for j, (top_a, right_a, bottom_a, left_a) in enumerate(
                            face_locations_augmented):
                        face_roi_augmented = augmented_image[top_a:bottom_a, left_a:right_a]
                        cropped_augmented_filename = (
                            f"{os.path.splitext(augmented_filename)[0]}_face_{j}.jpg"
                        )
                        cropped_augmented_path = os.path.join(
                            output_dir, cropped_augmented_filename
                        )
                        cv2.imwrite(cropped_augmented_path, face_roi_augmented)


def augment_image(image, flip, brightness_factor, scale_factor, rotation_angle, contrast_factor):
    if flip:
        image = cv2.flip(image, 1)

    # Brightness and contrast
    image = cv2.convertScaleAbs(image, beta=brightness_factor * 128, alpha=contrast_factor)

    # Scaling
    if scale_factor != 1.0:
        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    # Rotation
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

    return image

# --- Configuration ---
input_image_dir = "trained_dataset/simran"
output_image_dir = "cropped_data" 

# --- Augmentation Settings ---
flip_probability = 0.5
brightness_range = (0.5, 1.5)   
scale_range = (0.7, 1.3)         
rotation_range = (-30, 30)       
contrast_range = (0.7, 1.3)     
num_augmentations = 10             

# --- Run Augmentation ---
augment_and_save_faces(input_image_dir, output_image_dir,
                      flip_probability=flip_probability,
                      brightness_range=brightness_range,
                      scale_range=scale_range,
                      rotation_range=rotation_range,
                      contrast_range=contrast_range,
                      num_augmentations=num_augmentations)