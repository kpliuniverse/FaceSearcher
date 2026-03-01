import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import pathlib
import threading

import PIL
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import wx.lib.scrolledpanel
import wx

logging.basicConfig(level=logging.INFO)
FACE_CLASSIFIER = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


def ssim_two_images(image1: Image, image2: Image):
    """
    Compares two same-sized images using Structural Similarity Index (SSIM).
    """
    gray1 = image1.convert("L")
    gray2 = image2.convert("L")
    return ssim(np.array(gray1), np.array(gray2), gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)


def compare_images(image1: cv2.Mat, image2: cv2.Mat):
    """
    Prepares two images for comparison and compares them using SSIM.
     
    This includes resizing the images to the same size and converting them to RGB format.
    """
    image1 = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    image2 = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    w = min(image1.size[0], image2.size[0])
    h = min(image1.size[1], image2.size[1])
    image1 = image1.resize((w, h))
    image2 = image2.resize((w, h))
    return ssim_two_images(image1, image2)


def get_analysis(image_path: str, dest_folder: pathlib.Path, threshold: float, threads: int): 
    """
    Compare source image with all images in the destination folder.
    
    This returns a list of tuples of (image_path, similarity) sorted by similarity in descending order. 
    Only the first face detected in the source image will be used for comparison.
    Nesting is supported for the destination folder, but only images with faces will be compared.
    """   
    image = cv2.imread(image_path)
    faces = detect_faces(image)
    if len(faces) == 0:
        logging.info("No faces detected at source image.")
        return []
    face = faces[0]
    logging.info(f"Detected {len(faces)} faces at source image, only using the first one.")
    del faces

    x_source, y_source, w_source, h_source = face
    cropped_source_image = image[y_source:y_source+h_source, x_source:x_source+w_source]
    similiarity_data = dict()

    paths  = collections.deque([dest_folder])
    cur_path = cur_path = paths.popleft()

    image_comparison = ImageComparison()

    while cur_path:         
        with ThreadPoolExecutor(max_workers=threads) as executor:
            for image_path in os.listdir(cur_path):
                dest_path = pathlib.Path(cur_path / image_path)
                if dest_path.is_dir():
                    paths.append(dest_path)
                    continue
                logging.info(f"Comparing with {dest_path}...")
                executor.submit(image_comparison.compare_faces, cropped_source_image, dest_path, threshold)
            cur_path = paths.popleft() if len(paths) > 0 else None
    
    return sorted(image_comparison.result.items(), key=lambda x: x[1], reverse=True)

# Source - https://stackoverflow.com/a/21378718
# Posted by Jerry_Y, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-26, License - CC BY-SA 4.0

class GUI(wx.Frame):

    def __init__(self, data):
        #First retrieve the screen size of the device
        screenSize = (800, 450)    
        screenWidth, screenHeight = screenSize
        #Create a frame
        wx.Frame.__init__(self,None,-1,"Image Similiarity Result",size=screenSize, style=wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER)
   
        panel2 = wx.lib.scrolledpanel.ScrolledPanel(self,-1, size=(screenWidth,400), pos=(0,28), style=wx.SIMPLE_BORDER)
        panel2.SetupScrolling()
        panel2.SetBackgroundColour('#FFFFFF')
        bSizer = wx.BoxSizer( wx.VERTICAL )
        
       
        for path, similiarity in data:
            rowsizer = wx.BoxSizer( wx.HORIZONTAL )
            row =  wx.Panel(panel2,size=(screenWidth,255), pos=(0,0), style=wx.SIMPLE_BORDER)
            icon = PIL.Image.open(path)
            resized_icon = icon.resize((250, 250))
            wx_image = wx.Image(250, 250, np.array(resized_icon))
            bitmap = wx_image.ConvertToBitmap()
            bitmap_widget = wx.StaticBitmap(parent=row, bitmap=bitmap)
            rowsizer.Add(bitmap_widget, 0, wx.ALL, 5)
            bSizer.Add( row, 0, wx.EXPAND, 5 )
            icon.close()
            st = wx.StaticText(row, id = 1, label =f"{path}\nSimiliarity: {similiarity}", pos =(0, 0), size = wx.DefaultSize, style = 0)
            rowsizer.Add(st, 0, wx.ALL, 5)
            row.SetSizer(rowsizer)  
            
        panel2.SetSizer( bSizer )


def show_result(data):
    """
    Show face search results to a GUI.
    """
    app = wx.App()
    frame = GUI(data)
    frame.Show()
    app.MainLoop()


def detect_faces(image: cv2.Mat):
    """
    Get faces of a particular image using OpenCV's Haar Cascade Classifier.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    faces = FACE_CLASSIFIER.detectMultiScale(
        gray_image, scaleFactor=1.25, minNeighbors=6, minSize=(50, 50)
    )
    return faces


class ImageComparison:
    def __init__(self, balance=0):
        self.result = dict()
        self.dict_lock = threading.Lock()

    def detect_faces(self, image: cv2.Mat):
        """
        Get faces of a particular image using OpenCV's Haar Cascade Classifier.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        with self.dict_lock:
            faces = FACE_CLASSIFIER.detectMultiScale(
                gray_image, scaleFactor=1.25, minNeighbors=6, minSize=(50, 50)
            )
        return faces

    def compare_faces(self, cropped_source_image, dest_path, threshold):        
        """
        Compare a cropped source face with faces detected in the destination image, 
        and add the result to the result dicationary.
        """
        try:
            dest_image = cv2.imread(dest_path)
        except Exception as e:
            logging.error(f"Error reading {dest_path}: {e}")
            return
        try:
            dest_faces = self.detect_faces(dest_image)
        except Exception as e:
            logging.error(f"Error detecting faces in {dest_path}: {e}")
            return
        try:
            if len(dest_faces) == 0:
                logging.info(f"No faces detected at destination image {dest_path}, skipping.")
                return
            for i, dest_face in enumerate(dest_faces):
                x_dest, y_dest, w_dest, h_dest = dest_face
                cropped_dest_image = dest_image[y_dest:y_dest+h_dest, x_dest:x_dest+w_dest]
                similarity = compare_images(cropped_source_image, cropped_dest_image)
                if similarity < threshold:
                    logging.info(f"Similarity {similarity} below threshold {threshold} on face {i} on image {dest_path}, skipping.")
                    continue
                with self.dict_lock:
                    self.result[dest_path] = similarity
        except Exception as e:
            logging.error(f"Error comparing face with {dest_path}: {e}")

            
def main(args):
    print(cv2.__file__)
    parser = argparse.ArgumentParser(prog="FaceSearcher", description="Compare faces in source image with a set of images in the destination folder.")
    parser.add_argument("-s", "--source_image", help="Path to the source image.")
    parser.add_argument("-c", "--comparison_folder", help="Path to the comparison folder. Nesting supported.")
    parser.add_argument("--threshold", help="Similarity threshold, between 0 and 1, default is 0.3.", default=0.3, type=float) 
    parser.add_argument("--threads", help="Number of threads to use for face comparison, default is 4.", default=4, type=int)

    args = parser.parse_args(args[1:])
    if not args.source_image or not args.comparison_folder:
        parser.print_help()
        return 0
    dest_path = pathlib.Path(args.comparison_folder)
    if data := get_analysis(args.source_image, dest_path, args.threshold, args.threads):
        show_result(data)