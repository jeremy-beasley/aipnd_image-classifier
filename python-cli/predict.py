"""
-------------------------------------
FILE:       predict.py

AUTHOR:     Jeremy Beasley 
EMAIL:      github@jeremybeasley.com
CREATED:    20190727
REVISED:    20190728

PURPOSE:    To make prediction using a pre-trained NN and display the top-k classes inferrer from input image

Example calls: 
    python predict.py ./flowers/test/101/image_07952.jpg checkpoint.pth --gpu
-----------------------------------
"""

# ----------------------------------------------
# -------------------- IMPORTS -----------------
# ----------------------------------------------

import optparse
import time 
import json
import network as cnn
import util

# ----------------------------------------------
# -------------------- PARSE INPUT -------------
# ----------------------------------------------

parser = optparse.OptionParser("predict.py image_directory checkpoint [Options]")
parser.add_option('--top_k', action="store", dest="top_k", default=5, type="int", help="Return K-most likely classes")
parser.add_option('--category_names', action="store", dest="category_names", default="cat_to_name.json", type="str", help="Mapping file for categories to labels")
parser.add_option('--gpu', action="store_true", dest="gpu", default=True, help="Use GPU for training")

# parse all arguments
options, args = parser.parse_args()

# break: in case there are no command line params
if len( args ) < 2:
    parser.error("Usage: python predict.py image_directory checkpoint")

# set data_directory from the argument line
param_image_file = args[0]                # default: ./flowers/test/101/image_07952.jpg
param_load_file_name = args[1]            # default: checkpoint.pth
param_output_size = 102                   # 102 - original # 10 - test


print("---- Running with parameters ----")
print("----------------------------------")
print("image file:         ", param_image_file)
print("load file:          ", param_load_file_name)

if options.top_k is not None:
    print("top k:              ", options.top_k)
    
if options.category_names is not None:
    print("category names:     ", options.category_names)
    
if options.gpu is not None:
    print("gpu:                ", options.gpu)
print("-------------------------------")


# ------ Load mapping dictionary -----------
print("Loading mapping dictionary ... ", end="")
with open(options.category_names, 'r') as f:
    cat_to_name = json.load(f)
print("done")

def main(): 
    # ------ Load trained model -----------
    model = util.load_model(param_load_file_name, options.gpu)

    # ------ Make inference -----------
    print("Getting prediction ... ", end="")
    top_ps, top_labels, top_flowers = model.predict(param_image_file, options.top_k, cat_to_name)

    for i in range( len(top_flowers) ):
        print(" {} with {:.3f} is {}".format(i+1, top_ps[i], top_flowers[i]))
    print("done")


# util.imshow(util.process_image(param_image_file))
# util.plot_bar(top_ps, top_flowers)

# --- Call to main function --------- 
if __name__ == "__main__": 
    main()
