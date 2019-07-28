"""
-------------------------------------
FILE: train.py

AUTHOR:     Jeremy Beasley 
EMAIL:      github@jeremybeasley.com
CREATED:    20190727
REVISED:    20190728

PURPOSE:    To build, train and save a NN capabable of classifying a directory of images


Example calls: 
    python train.py flowers --gpu
-----------------------------------
"""

# ----------------------------------------------
# -------------------- IMPORTS -----------------
# ----------------------------------------------

import optparse
import time 
import network as cnn
import util

# ----------------------------------------------
# -------------------- PARSE INPUT -------------
# ----------------------------------------------

parser = optparse.OptionParser("train.py data_directory [Options]")
parser.add_option('--save_dir', action="store", dest="save_directory", default="./", type="str", help="Set path to save checkpoints")
parser.add_option('--arch', action="store", dest="architecture", default="vgg13", type="str", help="Choose architecture, e.g. 'vgg13'")
parser.add_option('--learning_rate', action="store", dest="learning_rate", default=0.001, type="float", help="Set hyperparameter: learning rate (0.001)")
parser.add_option('--hidden_units', action="store", dest="hidden_units", default=500, type="int", help="Set hyperparameter: hidden units (512)")
parser.add_option('--epochs', action="store", dest="epochs", default=2, type="int", help="Set hyperparameter: epochs (20)")
parser.add_option('--gpu', action="store_true", dest="gpu", default=False, help="Use GPU for training")

options, args = parser.parse_args() 

# --- Throw error is there are no parameters --------- 
if len(args) < 1: 
    parser.error("Usage: python train.py data_directory")

# --- Update input directory --------- 
param_input_path = args[0]          # default: flowers
param_output_size = 102             # 102 classes in flowers dataset
param_print_every = 20              # 20
param_model_save_filename = "checkpoint.pth"

print("---- Running with parameters ----")
print("----------------------------------")
print("input path: ", param_input_path)


# --- Check that parameters are set --------- 
if options.save_directory is not None:
    print("save directory: ", options.save_directory)
    
if options.architecture is not None:
    print("architecture:   ", options.architecture)
    
if options.learning_rate is not None:
    print("learning rate:  ", options.learning_rate)
    
if options.hidden_units is not None:
    print("hidden units:   ", options.hidden_units)
    
if options.epochs is not None:
    print("epochs:         ", options.epochs)

if options.gpu is not None:
    print("gpu:            ", options.gpu)
    param_device = "cuda" if options.gpu else "cpu"
print("-------------------------------")


def main(): 

    # ------- Create network ---------------------------
    model = cnn.Network(param_input_path, param_output_size, options.architecture, options.hidden_units, options.learning_rate, param_device)

    # --- Calculate time to train --------- 
    start = time.time()
    model.learn(options.epochs, param_print_every, param_device) 
    print("Time to learn: ", util.get_duration(time.time() - start))


    # --- Test the network --------- 
    start = time.time() 
    model.test(param_device)
    print("Time to test: ", util.get_duration(time.time() - start))

    # --- Save trained network --------- 
    model.save(options.architecture, options.hidden_units, param_output_size, options.learning_rate, options.epochs, options.save_directory + param_model_save_filename, param_input_path)
    

# --- Call to main function --------- 
if __name__ == "__main__": 
    main()