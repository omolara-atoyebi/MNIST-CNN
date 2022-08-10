import matplotlib.pyplot as plt 
import torch 
import torch.nn.functional as F
import data_handler as dh
#from train import model, trainloader, testloader
from model import CNN
import numpy as np

model = CNN()
state_dict = torch.load('model.pth')
model.load_state_dict(state_dict)
model.eval()


trainloader, testloader = dh.data_handler()





def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array(0.485,)
        std = np.array(0.229,)
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


def view_classify_general(img, ps, class_list):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    imshow(img, ax=ax1, normalize=True)
    ax1.axis('off')
    ax2.barh(np.arange(len(class_list)), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(len(class_list)))
    ax2.set_yticklabels([x for x in class_list], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

image, label = next(iter(testloader))
img, label = image[0], label[0]
# Flatten images
# Forward pass, get our logits
logits = model(img.view(1, *image[0].shape))
# Calculate the loss with the logits and the labels
ps = F.softmax(logits, dim=1)
    
m = view_classify_general(img, ps, class_list= [0,1,2,3,4,5,6,7,8,9])
print(m)




