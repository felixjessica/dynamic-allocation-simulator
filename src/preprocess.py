"""
Utility file
utility functions like loading data and preprocessing images.
"""

import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np


def process_image(image_path):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    """

    # TODO: Process a PIL image for use in a PyTorch model

    # Define how to the transform should happen
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

    # Open image and transform it
    im = transform(Image.open(image_path))

    # Return only the transformed image as a Tensor, since imshow will convert to Numpy array
    return im


#######################################################################################################################

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension

    new_image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    new_image = std * new_image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    new_image = np.clip(new_image, 0, 1)

    ax.imshow(new_image)

    return ax

#######################################################################################################################

