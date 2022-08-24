import torch
import torch.nn as nn
import torch.nn.functional as F
import data_handler as dh
from model import CNN
import matplotlib.pyplot as plt


trainloader, testloader = dh.data_handler()
epochs = 50
# print(type(trainloader))
# print(type(testloader))

model = CNN()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
criterion = nn.CrossEntropyLoss()

train_losses = []
test_losses = []
accuracies = []
previous_accuracy = 0

for epoch in range(epochs):
    tr_loss = 0
    for  images,labels in iter(trainloader):
        #img = images.reshape(images.shape[0], -1)

        #print(images.shape)

        optimizer.zero_grad()
        prediction = model.forward(images)
        train_loss= criterion(prediction, labels)
        train_loss.backward()
        optimizer.step()
        tr_loss += train_loss.item()
    trn_loss = tr_loss/len(trainloader)
    train_losses.append(trn_loss)


    model.eval()
    tst_loss = 0
    running_accuracies = 0
    with torch.no_grad():
        for image, label in iter(testloader):
            #imgs = image.view(image.shape[0],-1)
            Test_pred = model.forward(image)
            test_loss = criterion(Test_pred,label)
            tst_loss +=  test_loss.item()

            classes = F.softmax(Test_pred,dim=1).argmax(dim=1)
            accuracy = sum(classes == label)/len(Test_pred)
            running_accuracies += accuracy.item()
        tes_loss = tst_loss/len(testloader)
        accuracy1 = running_accuracies/len(testloader)
        test_losses.append(tes_loss)
        accuracies.append(accuracy1)
        print(f" Loss: {tr_loss:.4f}  test_loss:  {tst_loss:.4f}   accuracy: {accuracy1 *100: .2f}")
    if   accuracy1 > previous_accuracy:
        
        torch.save(model.state_dict(), f'model_{accuracy1:.2f}.pth')

        previous_accuracy = accuracy1

    model.train()





x_epoch = list(range(epochs))
plt.figure(figsize=(15, 6))
plt.plot(x_epoch ,train_losses, label='Train loss')
plt.plot(x_epoch ,test_losses, label='Test loss')
plt.plot(x_epoch ,accuracies, label='Accuracy')
plt.legend()
plt.show()


    


