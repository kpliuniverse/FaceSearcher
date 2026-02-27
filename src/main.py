import argparse
import collections
import logging
import os
import pathlib

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


def get_analysis(image_path: str, dest_folder: pathlib.Path, threshold: float = 0.3):    
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
    
    return sorted(similiarity_data.items(), key=lambda x: x[1], reverse=True)

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
            st = wx.StaticText(row, id = 1, label =f"{path}\nSimiliarity: {similiarity}", pos =(0, 0), size = wx.DefaultSize, style = 0)
            rowsizer.Add(st, 0, wx.ALL, 5)
            row.SetSizer(rowsizer)  
            
        panel2.SetSizer( bSizer )


def show_result(data):
    app = wx.App()
    frame = GUI(data)
    frame.Show()
    app.MainLoop()


def main(args):
    parser = argparse.ArgumentParser(prog="FaceSearcher", description="Compare faces in source image with a set of images in the destination folder.")
    parser.add_argument("-s", "--source_image", help="Path to the source image.")
    parser.add_argument("-c", "--comparison_folder", help="Path to the comparison folder. Nesting supported")
    parser.add_argument("--threshold", help="Similarity threshold, between 0 and 1, default is 0.3.", default=0.3, type=float)
    args = parser.parse_args(args[1:])
    dest_path = pathlib.Path(args.comparison_folder)
    if data := get_analysis(args.source_image, dest_path, args.threshold):
        show_result(data)