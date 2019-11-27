import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from modelos import NNh2_relu
import matplotlib.pyplot as plt


def precision(prueba=True):
  loader = test_loader if prueba else train_loader
  total=0
  correct=0
  with torch.no_grad():
    for xi, yi in loader:
      xi = xi.reshape(-1, 28*28).to(device)
      yi = yi.to(device)
      output = model(xi)
      _, predicted = torch.max(output.data, 1)
      total += yi.size(0)
      correct += (predicted == yi).sum().item()
  return correct/total

def perdida(prueba=True):
  trainset_loss = 0.0
  loader = test_loader if prueba else train_loader
  with torch.no_grad():
    for xi, yi in loader:
      xi = xi.reshape(-1, 28*28).to(device)
      yi = yi.to(device)
      output = model(xi)
      loss = loss_function(output, yi)

    batch_loss = loss.item() * yi.size(0)
    trainset_loss += batch_loss
  return trainset_loss

device = torch.device('cuda'if torch.cuda.is_available() else'cpu')

input_size = 784 # Dimension de datos de entrada (28 x 28)
num_classes = 10 # MNIST tiene 10 clases (numeros del 1 al 10)
num_epochs = 10 # Numero de epocas para entrenar
bs = 100 # Tamano de lote (batch_size)

root ='./datos'# Carpeta donde se guardaran los datos
train_data = MNIST(root, train=True, transform=ToTensor(), download=True)
test_data  = MNIST(root, train=False, transform=ToTensor())

train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=bs, shuffle=False)

lr = 0.01 # Tasa de aprendizaje
model = NNh2_relu(input_size, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1,-1)

perdida_mod = []
precision_mod = []
for epoch in range(num_epochs):

  for i, (xi, yi) in enumerate(train_loader):

    # Las entradas de la imagen se convierten en vectores
    xi = xi.reshape(-1, 28*28).to(device)# imagenes
    yi = yi.to(device)# etiquetas

    # Propagacion para adelante
    output = model(xi)
    loss = loss_function(output, yi)
    # Propagcion para atras y paso de optimizacion

    scheduler.step()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #perdida_mod.append(loss.item())
    #if (i+1) % 100 == 0:
    #  print ('Epoca: {}/{}, Paso: {}/{}, Perdida: {:.5f}'.format(epoch+1,num_epochs, i+1, len(train_loader), loss.item()))

  perdida_mod.append(perdida())
  precision_mod.append(precision())


plt.subplot(1,2,1)
plt.plot(perdida_mod)
plt.title('Pérdida')
plt.subplot(1,2,2)
plt.plot(precision_mod)
plt.title('Precisión')
plt.show()
