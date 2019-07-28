"""
-------------------------------------
file: util.py

author:     Jeremy Beasley 
email:      github@jeremybeasley.com
created:    20190727
updated:    20190728
-----------------------------------
"""

# ----------------------------------------------
# -------------------- IMPORTS -----------------
# ----------------------------------------------

import matplotlib.pyplot as plt

from PIL import Image

import numpy as np 
import time
import datetime
import json

import torch
from torchvision import transforms

import network as cnn



# ----------------------------------------------
# ----------- HELPER FUNCTIONS -----------------
# ----------------------------------------------

def load_model(filename, device): 
    """ Load model """
    
    print("Loading trained model from: {} ... ".format(filename))
    checkpoint = torch.load(filename) 
    model = cnn.Network(checkpoint['data_directory'], checkpoint['output_size'], checkpoint['arch'], checkpoint['hidden_units'], checkpoint['learning_rate'], device)
    # model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dictionary(checkpoint['state_dict'])
    print("done!")
    
    return model



def get_duration(duration): 
    """ Calculate the duration in hh::mm::ss """
    seconds = int(duration%60)
    minutes = int(duration/60)%60
    hours   = int(duration/3600)%24
    
    output = "{:0>2}:{:0>2}:{:0>2}".format(hours, minutes, seconds)
    return output


def get_current_time():
    """ Return the current date, time """
    
    # --- Get UTC time and then convert to local time --------- 
    utc_dt = datetime.datetime.now(datetime.timezone.utc)
    dt = utc.astimezone()
    return str(dt)



def plot_bar(np_ps, np_flower_names):
    """ Plot a bar graph """
    
    y_pos = np.arange(len(np_flower_names))
    
    plt.barh(y_pos, np_ps, align='center', alpha=0.5)
    plt.yticks(y_pos, np_flower_names)
    plt.gca().invert_yaxis()   # invert axis to show the highest probabiliy at the top position
    plt.xlabel("Probability from 0.0 to 1.0")
    plt.title("Flower")
    plt.show()
    
    
    
def process_image(image):
    ''' Processes a PIL image—scales, crops, and normalizes—for a PyTorch model,
        returns an Numpy array
    '''
    
    print("Loading image data ... ", end="")
    prediction_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])
    
    img_pil = Image.open(image)
    img_tensor = prediction_transforms(img_pil)
    print("done!")
    return img_tensor.numpy()



def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax