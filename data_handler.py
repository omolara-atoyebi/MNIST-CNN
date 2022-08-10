from torchvision import datasets,transforms
import torch 


def data_handler():
    train_transform = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))])
    test_transform = transforms.Compose([ transforms.ToTensor(),
                         transforms.Normalize((0.5,),(0.5,))])
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    

    testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)



    return trainloader, testloader


    
