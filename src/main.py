import logging
import os

import PIL
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image


logging.basicConfig(level=logging.INFO)
FACE_CLASSIFIER = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

def detect_faces(image: cv2.Mat):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    faces = FACE_CLASSIFIER.detectMultiScale(
        gray_image, scaleFactor=1.25, minNeighbors=6, minSize=(50, 50)
    )
    return faces

def ssim_two_images(image1: Image, image2: Image):
    gray1 = image1.convert("L")
    gray2 = image2.convert("L")
    return ssim(np.array(gray1), np.array(gray2))

def compare_images(image1: cv2.Mat, image2: cv2.Mat):
    image1 = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    image2 = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    print(image1.size, image2.size)

    return image1, image2
def main(args):
    image = cv2.imread(args[1])
    faces = detect_faces(image)
    if len(faces) == 0:
        logging.info("No faces detected at source image.")
        return 0
    face = faces[0]
    logging.info(f"Detected {len(faces)} faces at source image, only using the first one.")
    del faces

    x_source, y_source, w_source, h_source = face
    cropped_source_image = image[y_source:y_source+h_source, x_source:x_source+w_source]
    for image_path in (os.listdir("data/")):
        os.path.join("data/", image_path)
        
        logging.info(f"Comparing with {image_path}...")
        dest_image = cv2.imread(os.path.join("data/", image_path))
        for dest_face in detect_faces(dest_image):
            x_dest, y_dest, w_dest, h_dest = dest_face
            cropped_dest_image = dest_image[y_dest:y_dest+h_dest, x_dest:x_dest+w_dest]
            similarity = compare_images(cropped_source_image, cropped_dest_image)