"""
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass
in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint
Options:
Return top-k most likely classes:               python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names:      python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference:                          python predict.py input checkpoint --gpu
"""

# TODO: IMPORTS HERE
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import torch
import argparse
import utility as util
import json

#######################################################################################################################

# TODO: Add command line arguments ######

# Set DEFAULT values for all future arguments
default_checkpoint = 'checkpoint.pth'
default_cat_name_file = 'cat_to_name.json'
default_image_path = 'flowers/test/10/image_07104.jpg'
default_top_k = 5
default_arch = 'densenet161'
default_device = 'gpu'

# Creates Argument Parser object called parser
parser = argparse.ArgumentParser()

# Arguments for inputs
parser.add_argument('--checkpoint', type=str, default=default_checkpoint, help='Name of checkpoint of model for '
                                                                               'prediction')
parser.add_argument('--arch', type=str, default=default_arch, help='Choose between "densenet161" or "alexnet"')
parser.add_argument('--dir', type=str, default=default_image_path, help='File path of image to predict')
parser.add_argument('--top_k', type=int, default=default_top_k, help='Number of classes in the prediction output')
parser.add_argument('--json', type=str, default=default_cat_name_file, help='File with names of categories')
parser.add_argument('--gpu', type=str, default=default_device, help='Uses "GPU" if available')

# Assign variable in parse_args() to access the arguments in the argparse object
args = parser.parse_args()

# Assign the variables to be used in the rest of this script
checkpoint = args.checkpoint
cat_name_file = args.json
image_path = args.dir
top_k = args.top_k
arch = args.arch
device = args.gpu

#######################################################################################################################

# TODO: Open the category to name file json file
with open(cat_name_file, 'r') as f:
    cat_to_name = json.load(f)


#######################################################################################################################
# TODO: Load Model/Checkpoint

def load_checkpoint(filepath):
    """
    Inputs: filepath to the checkpoint (.pth) extension
    Outputs: model.state_dict -- model architecture info, includes parameter matrices for each of the layers
             optimizer -- optimizer parameters
             model.class_to_idx --
    """

    # Load the checkpoint
    checkpoint_load = torch.load(filepath)

    # Load model architecture and parameters
    model.classifier = checkpoint_load['classifier']
    model.load_state_dict(checkpoint_load['state_dict'])
    model.class_to_idx = checkpoint_load['class_to_idx']

    # Load hyper-parameters (optional)
    epochs = checkpoint_load['epochs']
    learning_rate = checkpoint_load['learning_rate']

    # Load other relevant information (optional)
    network = checkpoint_load['network']
    input_size = checkpoint_load['input_size']
    output_size = checkpoint_load['output_size']
    hidden_units = checkpoint_load['hidden_units']
    optimizer = checkpoint_load['optimizer']

    print(f"Model - Classifier: {model.classifier}\n"
          f"Learning rate: {learning_rate}\n"
          f"Network: {network}\n"
          f"Epochs: {epochs}\n"
          f"Input size: {input_size}\n"
          f"Output size: {output_size}\n"
          f"Hidden layers: {hidden_units}\n"
          f"Optimizer: {optimizer}\n")

    return model


# DEFINE THE MODEL DURING THE LOAD
model = load_checkpoint(checkpoint)


#######################################################################################################################
# TODO: Define prediction function

def predict(fn_image_path, fn_model, fn_topk):
    """ Predict the class (or classes) of an image using a trained deep learning model.
        Inputs: image_path, model, topk (default=5)
        Outputs: top-k probabilities and top-k classes
    """

    # TODO: Implement the code to predict the class from an image file

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Process any input image using pre-defined function: process_image(image_path)
    image = util.process_image(image_path)

    # to accommodate RuntimeError error
    image = image.unsqueeze(0)

    # Move image to 'device'
    image = image.to(device)

    # Predict the image
    # Calculate log(Softmax) -> Convert to Prob distribution
    log_ps = model(image)
    ps = torch.exp(log_ps)

    # Get top-k probabilities
    top_prob, top_class = ps.topk(k=top_k, dim=1)

    # Move top prob to cpu (vs cuda), then numpy (vs tensor), then into a list
    top_prob = top_prob.detach().cpu().numpy().tolist()[0]

    # Convert top labels to cpu (vs cuda), then numpy (vs tensor), then into a list
    top_class = top_class.detach().cpu().numpy()[0]

    # Invert class_to_idx --> idx_to_class and save to dictionary
    idx_to_class = {idx: cls for cls, idx in model.class_to_idx.items()}

    # Assign labels to the top_class
    class_labels = [idx_to_class[cls] for cls in top_class]

    flowers = [cat_to_name[label] for label in class_labels]

    # return probability of top_k and their corresponding class names
    return top_prob, class_labels, flowers


# TODO: Display topK with their probabilities
prob, classes, flower_names = predict(fn_image_path=image_path, fn_model=model, fn_topk=top_k)
print(prob)
print(classes)
print(flower_names)

#######################################################################################################################
