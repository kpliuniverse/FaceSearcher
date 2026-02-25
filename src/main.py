import collections
import logging
import os
import pathlib

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
    return ssim(np.array(gray1), np.array(gray2), gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)

def compare_images(image1: cv2.Mat, image2: cv2.Mat):
    image1 = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    image2 = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    w = min(image1.size[0], image2.size[0])
    h = min(image1.size[1], image2.size[1])
    image1 = image1.resize((w, h))
    image2 = image2.resize((w, h))

    return ssim_two_images(image1, image2)

def main(args):
    threshold = 0.3
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
    similiarity_data = dict()

    paths  = collections.deque([pathlib.Path("data/")])
    cur_path = cur_path = paths.popleft()
    while cur_path:            
        for image_path in os.listdir(cur_path):
            dest_path = pathlib.Path(cur_path / image_path)
            
            if dest_path.is_dir():
                paths.append(dest_path)
                continue
            logging.info(f"Comparing with {dest_path}...")
            dest_image = cv2.imread(dest_path)
            dest_faces = detect_faces(dest_image)
            if len(dest_faces) == 0:
                logging.info(f"No faces detected at destination image {image_path}, skipping.")
                continue
            for dest_face in dest_faces:
                x_dest, y_dest, w_dest, h_dest = dest_face
                cropped_dest_image = dest_image[y_dest:y_dest+h_dest, x_dest:x_dest+w_dest]
                similarity = compare_images(cropped_source_image, cropped_dest_image)
                if similarity < threshold:
                    logging.info(f"Similarity {similarity} below threshold {threshold}, skipping.")
                    continue
                similiarity_data[dest_path] = similarity
        cur_path = paths.popleft() if len(paths) > 0 else None

