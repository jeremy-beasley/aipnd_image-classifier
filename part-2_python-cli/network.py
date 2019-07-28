"""
-------------------------------------
file: network.py

author:     Jeremy Beasley 
email:      github@jeremybeasley.com
created:    20190727
updated:    20190728
-----------------------------------
"""

# ----------------------------------------------
# -------------------- IMPORTS -----------------
# ----------------------------------------------

import torch
from torch import nn, optim 
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from collections import OrderedDict
import util



# ----------------------------------------------
# -------------------- NN CLASS -------------------
# ----------------------------------------------
class Network(nn.Module): 
    def __init__(self, param_input_path, param_output_size, param_model_architecture, param_hidden_units, param_learning_rate, param_device):
        super(Network, self).__init__()
        print("CNN ... ") 
        print("Loading image data ... ", end="")

        # --- Define transforms for datasets --------- 
        self.train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.RandomVerticalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

        self.test_transforms = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

        # --- Load the data --------- 
        self.train_data = datasets.ImageFolder(param_input_path + '/train', transform=self.train_transforms)
        self.valid_data = datasets.ImageFolder(param_input_path + '/valid', transform=self.test_transforms)
        self.test_data = datasets.ImageFolder(param_input_path + '/test', transform=self.test_transforms)

        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=64, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_data, batch_size=64)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=32)

        print("done")
        
        
        self.class_to_idx = self.test_data.class_to_idx
        
        print("Creating model ... ", end="")
        self.model = models.__dict__[param_model_architecture](pretrained=True)

        # --- Get in_features for classifier --------- 
        self.in_features = self.get_in_features(param_model_architecture)

        # --- Freeze parameters for pre-trained network to avoid backprop --------- 
        for param in self.model.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(OrderedDict([
                                      ('do1', nn.Dropout()), 
                                      ('fc1', nn.Linear(self.in_features, param_hidden_units)),
                                      ('relu', nn.ReLU()),
                                      ('do2', nn.Dropout()),
                                      ('fc2', nn.Linear(param_hidden_units, param_output_size)),
                                      ('output', nn.LogSoftmax(dim=1))]))


        # --- Train model with pre-trained network; attach new classifier --------- 
        self.model.classifier = self.classifier
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=param_learning_rate)
        print("done!")
        
        if param_device=="cuda" and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self. model)

        
    def get_in_features(self, param_architecture): 
        """ Return in_features for a given model"""

        in_features = 0

        # --- based on architecture --------- 
        if "densenet" in param_architecture: 
            in_features = self.model.classifier.in_features

        if "vgg" in param_architecture: 
            in_features = self.model.classifier[0].in_features

        return in_features
    
    
    # ----------------------------------------------
    # --------------- BUILD & TRAIN ----------------
    # ----------------------------------------------

    def learn(self, epochs, print_every, device): 
        """ Trains a neural network model """

        print("Start learning on device {} ... ".format(device))


        epochs = epochs
        print_every = print_every
        steps = 0

        # --- Move model to appropriate device and put in training mode --------- 
        self.model.to(device)
        self.model.train()

        # --- Train model --------- 
        for e in range(epochs): 
            training_loss = 0
            for inputs, labels in self.train_loader: 
                steps += 1

                # --- Move inputs and label tensors to the appropriate device --------- 
                inputs, labels = inputs.to(device), labels.to(device)

                self.optimizer.zero_grad()

                # --- Forward and back propogation --------- 
                log_ps = self.model(inputs)
                loss = self.criterion(log_ps, labels)
                loss.backward()
                self.optimizer.step()

                training_loss += loss.item()

                # --- Validate and print output --------- 
                if steps & print_every == 0:
                    # --- Put network in eval mode to test inference --------- 
                    self.model.eval()

                    # --- Gradients unnecessary for validation --------- 
                    with torch.no_grad(): 
                        test_loss, accuracy = self.validate(device)

                        print("epoch: {}/{}.. ".format(e+1, epochs),
                              "training loss: {:.3f}.. ".format(training_loss/print_every),
                              "validation loss: {:.3f}.. ".format(test_loss/len(self.valid_loader)),
                              "validation accuracy: {:.3f}".format(accuracy/len(self.valid_loader)))

                    training_loss = 0

                    # --- Put network back in training mode for next batch --------- 
                    self.model.train()

        print("... Done!")


    def validate(self, device): 
        """ Performs model validation """

        test_loss = 0
        accuracy = 0

        # --- Move model to approrriate device --------- 
        self.model.to(device)

        for inputs, labels in self.valid_loader: 

            # --- Move inputs and label tensors to the appropriate device --------- 
            inputs, labels = inputs.to(device), labels.to(device)

            log_ps = self.model(inputs)
            test_loss += self.criterion(log_ps, labels).item()

            # --- Convert output to probabilities to compare labels --------- 
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        return test_loss, accuracy


    def test(self, device): 
        """ Function summary """

        print("Calculate testing accuracy ... ", end="")
        correct = 0
        total = 0

        # --- Move model to approrriate device --------- 
        self.model.to(device)

         # --- Put network in eval mode to test inference --------- 
        self.model.eval()

        with torch.no_grad():
            for inputs, labels in self.test_loader: 

                # --- Move inputs and label tensors to the appropriate device --------- 
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print("done.")
        print("Network accurary on test images: %d%%" % (100*correct/total))
        


    def load_state_dictionary(self, state_dictionary):
        ''' helper function to load_state_dict '''
        self.model.load_state_dict(state_dictionary)
        
        
    def save(self, architecture, hidden_units, output_size, learning_rate, epochs, filename, data_path):
        """ Save trained model for later use"""

        print("Saving model to: ", filename, end="")

        # --- Configure checkpoint ---------  
        checkpoint = {"arch": architecture,
                      "in_features": self.in_features, 
                      "hidden_units": hidden_units, 
                      "learning_rate": learning_rate, 
                      "output_size": output_size,
                      "data_directory": data_path,
                      "epochs": epochs,
                      "optimizer_state_dict": self.optimizer.state_dict,
                      "class_to_idx": self.class_to_idx,
                      "state_dict": self.model.state_dict()}
        torch.save(checkpoint, filename)
        print(" ... done!")


    def predict(self, image_path, topk, cat_to_name):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
            TODO — Modify docstrings
        '''

        # --- Load image to get prediction --------- 
        img_np = util.process_image(image_path)
        print("Getting prediction ... ", end="")

        # --- Convert image to tensor for prediction --------- 
        img_tensor = torch.from_numpy(img_np).type(torch.FloatTensor)
        img_tensor.unsqueeze_(0)

        # --- Get probabilities --------- 
        self.model.eval()

        with torch.no_grad():
            img_variable = Variable(img_tensor)
            log_ps = self.model(img_variable)

        ps = torch.exp(log_ps)
        top_ps, top_class = ps.topk(topk)
        top_ps = top_ps.detach().numpy().tolist()[0]
        top_class = top_class.detach().numpy().tolist()[0]

        # --- Convert indices to classes and invert --------- 
        idx_to_class = {val: key for key, val in self.class_to_idx.items()}

        top_labels = [idx_to_class[label] for label in top_class]
        top_flowers = [cat_to_name[idx_to_class[label]] for label in top_class]

        print("done!")
        return top_ps, top_labels, top_flowers