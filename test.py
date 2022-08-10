
from random import triangular
import data_handler as dh

trainloader , testloader = dh.data_handler()


#images, labels = next(iter(trainloader))

print(iter(trainloader).next())