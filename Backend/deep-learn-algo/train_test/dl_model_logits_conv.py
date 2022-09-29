""" dl_model_logits_conv.py: 
    
    This program is used to convert logits to softmax probabilities, for debugging purposes
"""

__author__      = "Breydon Verryt-Reid"
__date__        = "26 Sep 22"
__version__     = 1.0
__status__      = "Complete"
__notes__       = "This program is intended to be run as a standalone program"

# importing required libraries
from scipy.special import softmax
import torch

def logitsconv(logits):
    """Convert logits to softmax probabilities.
    
    ** Params **
        logits (torch.Tensor): Logits of shape (batch_size, num_classes).
    ** Returns **
        torch.Tensor: Softmax probabilities of shape (batch_size, num_classes).
    """
    return torch.softmax(logits, dim=1)

result = logitsconv(torch.tensor([-4.8514,  5.4992]))   
print(result)   