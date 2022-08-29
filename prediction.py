from ast import Bytes
from io import BytesIO
from statistics import mode
from tkinter import Image
from urllib import request
import matplotlib.pyplot as plt 
import torch 
import torch.nn.functional as F
import data_handler as dh
#from train import model, trainloader, testloader
from model import CNN
import numpy as np
import requests
import cv2
from torchvision import datasets,transforms

state_dict = torch.load('model_0.79.pth')
model = CNN()
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

    ax.imshow(image, cmap= 'gray')
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
    plt.show()

# image, label = next(iter(testloader))
# idx = np.random.randint(0,len(image),(10))
# for i in idx:
#     img = image[i]
#     #print(img.unsqueeze(0).shape)
#     #img=img.view(image[0].shape[0],-1)
#     logits = model(img.unsqueeze(0))
#     # Calculate the loss with the logits and the labels
#     ps = F.softmax(logits, dim=1)
#     #ps = torch.exp(logits, dim = 1)
        
#     view_classify_general(img, ps, class_list= [0,1,2,3,4,5,6,7,8,9])



imag = cv2.imread('handwritten_img2.png')
image = cv2.cvtColor(imag,cv2.COLOR_BGR2GRAY)

convert_tensor = transforms.ToTensor()
img = convert_tensor(image)

logits = model(torch.unsqueeze(img,axis=0).float())

ps = F.softmax(logits, dim=1)

view_classify_general(img, ps, class_list= [0,1,2,3,4,5,6,7,8,9])

        