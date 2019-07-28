#!/usr/bin/env python
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[ ]:


"""
-------------------------------------

AUTHOR:     Jeremy Beasley 
EMAIL:      github@jeremybeasley.com
CREATED:    20190727
REVISED:    20190728

-----------------------------------
"""


# In[108]:


# ----------------------------------------------
# -------------------- IMPORTS -----------------
# ----------------------------------------------

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

from PIL import Image

import numpy as np 
import time
import datetime
import json

import torch
from torch import nn, optim 
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from collections import OrderedDict
# --------------------


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[109]:


# ----------------------------------------------
# -------------------- INPUTS -------------------
# ----------------------------------------------


# --- Set training parameters ---------
param_input_path = "flowers"                    # default: flowers
param_train_path = param_input_path + "/train"  # ./train
param_valid_path = param_input_path + "/valid"  # ./valid
param_test_path = param_input_path + "/test"    # ./test
param_output_size = 102                         # 102 flower classes
param_model_save_filename = "checkpoint.pth"
param_model_save_path = "./"                    # ./
param_model_architecture = "vgg13"              # torchvision model architecture — densenet121, vgg19
param_learning_rate = 0.001                     # 0.001
param_hidden_units = 524                        # 524
param_epochs = 2                                # 2
param_print_every = 20                          # print every N steps
param_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Set device
# -------------------------------------


# TODO - in the future, change this to use better print formatting for a table
print("---- Training with parameters ----")
print("----------------------------------")
print("input path:         ", param_input_path)
print("training path:      ", param_train_path)
print("validation path:    ", param_valid_path)
print("test path:          ", param_test_path)
print("save path:          ", param_model_save_path)
print("architecture:       ", param_model_architecture) 
print("learning rate:      ", param_learning_rate) 
print("hidden units:       ", param_hidden_units) 
print("epochs:             ", param_epochs) 
print("device:             ", param_device) 
print("----------------------------------")


# ----------------------------------------------
# -------------- LOAD DATA ---------------------
# ----------------------------------------------

print("Loading image data ... ", end="")

# --- Define transforms for datasets --------- 
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# --- Load the data --------- 
train_data = datasets.ImageFolder(param_train_path, transform=train_transforms)
valid_data = datasets.ImageFolder(param_valid_path, transform=test_transforms)
test_data = datasets.ImageFolder(param_test_path, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

print("done")


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[110]:


# --- Load mapping dictionary --------- 
print("Loading mapping dictionary ... ", end="")
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
print("done")

class_to_idx = test_data.class_to_idx


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to
# GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.
# 
# **Note for Workspace users:** If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.

# In[123]:


# ----------------------------------------------
# --------------- BUILD & TRAIN ----------------
# ----------------------------------------------

def learn(model, train_loader, valid_loader, optimizer, criterion, epochs, print_every, device): 
    """ Trains a neural network model """
    
    print("Start learning on device {} ... ".format(device))
    
    epochs = epochs
    print_every = print_every
    steps = 0
    
    # --- Move model to appropriate device and put in training mode --------- 
    model.to(device)
    model.train()
    
    # --- Train model --------- 
    for e in range(epochs): 
        training_loss = 0
        for inputs, labels in train_loader: 
            steps += 1
            
            # --- Move inputs and label tensors to the appropriate device --------- 
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # --- Forward and back propogation --------- 
            log_ps = model(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item()
            
            # --- Validate and print output --------- 
            if steps & print_every == 0:
                # --- Put network in eval mode to test inference --------- 
                model.eval()
                
                # --- Gradients unnecessary for validation --------- 
                with torch.no_grad(): 
                    test_loss, accuracy = validate(model, valid_loader, criterion, device)
                    
                    print("epoch: {}/{}.. ".format(e+1, epochs),
                          "training loss: {:.3f}.. ".format(training_loss/print_every),
                          "validation loss: {:.3f}.. ".format(test_loss/len(valid_loader)),
                          "validation accuracy: {:.3f}".format(accuracy/len(valid_loader)))
                
                training_loss = 0
                
                # --- Put network back in training mode for next batch --------- 
                model.train()
                
    print("... Done!")


def validate(model, valid_loader, criterion, device): 
    """ Assesses accurary of model during training. Returns total loss and accuracy """    
    test_loss = 0
    accuracy = 0
    
    # --- Move model to approrriate device --------- 
    model.to(device)
    
    for inputs, labels in valid_loader: 
        
        # --- Move inputs and label tensors to the appropriate device --------- 
        inputs, labels = inputs.to(device), labels.to(device)
        
        log_ps = model(inputs)
        test_loss += criterion(log_ps, labels).item()
        
        # --- Convert output to probabilities to compare labels --------- 
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
    return test_loss, accuracy


def test(model, test_loader, device): 
    """ Calculate testing accurary of model. Prints accurary as percentage to console """
    
    print("Calculate testing accuracy ... ", end="")
    
    correct = 0
    total = 0
    
    # --- Move model to approrriate device --------- 
    model.to(device)
    
     # --- Put network in eval mode to test inference --------- 
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in test_loader: 
            
            # --- Move inputs and label tensors to the appropriate device --------- 
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    print("done.")
    print("Network accurary on test images: %d%%" % (100*correct/total))


# In[112]:


# ----------------------------------------------
# -------------- SAVE & LOAD MODEL ------------_
# ---------------------------------------------- 

def save_model(model, optimizer, class_to_idx, architecture, in_features, hidden_units, output_size, learning_rate, epochs, filename, data_path):
    """ Save trained model for later use. """
    
    print("Saving model to: ", filename, end="")
    
    # --- Configure checkpoint ---------  
    checkpoint = {"arch": architecture,
                  "in_features": in_features, 
                  "hidden_units": hidden_units, 
                  "learning_rate": learning_rate, 
                  "output_size": output_size,
                  "data_directory": data_path,
                  "epochs": epochs,
                  "optimizer_state_dict": optimizer.state_dict,
                  "class_to_idx": class_to_idx,
                  "state_dict": model.state_dict()}
    torch.save(checkpoint, filename)
    print(" ... done!")
    

    
def get_in_features(model, param_architecture): 
    """ Return in_features for a given model """
    
    in_features = 0
    
    # --- based on architecture --------- 
    if "densenet" in param_architecture: 
        in_features = model.classifier.in_features
    
    if "vgg" in param_architecture: 
        in_features = model.classifier[0].in_features
    
    return in_features



def load_model(filename, device): 
    """ Loads and returns model """
    
    print("Loading trained model from: {} ... ".format(filename))
    checkpoint = torch.load(filename) 
    
    print("Creating model ... ", end="")
    model = models.__dict__[checkpoint["arch"]](pretrained=True)
    
    # --- Freeze parameters for pre-trained networks to avoid backprop --------- 
    for param in model.parameters(): 
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                              ('do1', nn.Dropout()), 
                              ('fc1', nn.Linear(checkpoint['in_features'], checkpoint['hidden_units'])),
                              ('relu', nn.ReLU()),
                              ('do2', nn.Dropout()),
                              ('fc2', nn.Linear(checkpoint['hidden_units'], checkpoint['output_size'])),
                              ('output', nn.LogSoftmax(dim=1))]))
    
    model.classifier = classifier
    print("done!")
    
    print("Initializing model ... ", end="")
    model.load_state_dict(checkpoint['state_dict'])
    print("done!")
    
    class_to_idx = checkpoint['class_to_idx']
    
    return model, class_to_idx


# In[159]:


# ----------------------------------------------
# ----------- HELPER FUNCTIONS -----------------
# ----------------------------------------------

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


# In[114]:


# ----------------------------------------------
# ------------------ MAIN SCRIPT ---------------
# ---------------------------------------------- 

print("Creating model ... ", end="")
model = models.__dict__[param_model_architecture](pretrained=True)

# --- Get in_features for classifier --------- 
in_features = get_in_features(model, param_model_architecture)

# --- Freeze parameters for pre-trained network to avoid backprop --------- 
for param in model.parameters():
    param.requires_grad = False
    
classifier = nn.Sequential(OrderedDict([
                              ('do1', nn.Dropout()), 
                              ('fc1', nn.Linear(in_features, param_hidden_units)),
                              ('relu', nn.ReLU()),
                              ('do2', nn.Dropout()),
                              ('fc2', nn.Linear(param_hidden_units, param_output_size)),
                              ('output', nn.LogSoftmax(dim=1))]))


# --- Train model with pre-trained network; attach new classifier --------- 
model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=param_learning_rate)
print("done!")


# --- Calculate time to train --------- 
start = time.time()
learn(model, train_loader, valid_loader, optimizer, criterion, param_epochs, param_print_every , param_device) 
print("Time to learn: ", get_duration(time.time() - start))


# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[124]:


# --- Test the network --------- 
start = time.time() 
test(model, test_loader, param_device)
print("Time to test: ", get_duration(time.time() - start))


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[126]:


# --- Save trained network --------- 
save_model(model, optimizer, class_to_idx, param_model_architecture, in_features, param_hidden_units, param_output_size, param_learning_rate, param_epochs, param_model_save_path + param_model_save_filename, param_input_path)


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[142]:


# --- Load model --------- 
model, class_to_idx = load_model(param_model_save_path + param_model_save_filename, param_device)
criterion = nn.NLLLoss
optimizer = optim.Adam(model.classifier.parameters(), lr=param_learning_rate)
# model.parameters


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[143]:


def process_image(image):
    """ Processes a PIL image—scales, crops, and normalizes—for a PyTorch model. Returns a Numpy array. """
    
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


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[155]:


def imshow(image, ax=None, title=None):
    """ Imshow for Tensor """
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


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[156]:


def predict(image_path, model, topk=5):
    """ Predict the class (or classes) of an image using a trained deep learning model. """
    
    # --- Load image to get prediction --------- 
    img_np = process_image(image_path)
    print("Getting prediction ... ", end="")
    
    # --- Convert image to tensor for prediction --------- 
    img_tensor = torch.from_numpy(img_np).type(torch.FloatTensor)
    img_tensor.unsqueeze_(0)
    
    # --- Get probabilities --------- 
    model.eval()
    
    with torch.no_grad():
        img_variable = Variable(img_tensor)
        log_ps = model(img_variable)
        
    ps = torch.exp(log_ps)
    top_ps, top_class = ps.topk(topk)
    top_ps = top_ps.detach().numpy().tolist()[0]
    top_class = top_class.detach().numpy().tolist()[0]
    
    # --- Convert indices to classes and invert --------- 
    idx_to_class = {val: key for key, val in class_to_idx.items()}
    
    top_labels = [idx_to_class[label] for label in top_class]
    top_flowers = [cat_to_name[idx_to_class[label]] for label in top_class]
    
    print("done!")
    return top_ps, top_labels, top_flowers


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[161]:


# TODO: Display an image along with the top 5 classes

param_input_image = "./flowers/test/101/image_07952.jpg"
param_top_k = 5

print("Predictions")
top_ps, top_labels, top_flowers = predict(param_input_image, model, param_top_k)

for i in range(len(top_flowers)): 
    print(" {} with {:.3f} is {}".format(i+1, top_ps[i], top_flowers[i] ) )
# ----------------------------------

imshow(process_image(param_input_image))


# In[162]:


plot_bar(top_ps, top_flowers)


# In[ ]:




