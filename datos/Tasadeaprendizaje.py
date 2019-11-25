import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from modelos import Net

# Se verifica si esta disponible CUDA,en caso contraro se usa el procesador
device = torch.device('cuda'if torch.cuda.is_available() else'cpu')

# Hiper-parametros
input_size = 784 # Dimension de datos de entrada (28 x 28)
num_classes = 10 # MNIST tiene 10 clases (numeros del 1 al 10)
num_epochs = 5 # Numero de epocas para entrenar
bs = 100 # Tamano de lote (batch_size)
lr = 0.001 # Tasa de aprendizaje


# Hiper-parametros
input_size = 784 # Dimension de datos de entrada (28 x 28)
num_classes = 10 # MNIST tiene 10 clases (numeros del 1 al 10)
num_epochs = 10 # Numero de epocas para entrenar
bs = 100 # Tamano de lote (batch_size)
lr = 0.001 # Tasa de aprendizaje


# Se bajan los datos MINST

root ='./datos'# Carpeta donde se guardaran los datos
train_data = MNIST(root, train=True, transform=ToTensor(), download=True)
test_data  = MNIST(root, train=False, transform=ToTensor())

# Se crea el iterable sobre el dataset
train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=bs, shuffle=False)

# Se define el modelo 
model = Net(input_size, num_classes).to(device)
loss_function = nn.CrossEntropyLoss()


lr = 0.001 # Tasa de aprendizaje
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
# [COMENTARIO] Â¿Por que multiplicamos por yi.size(0)?
    batch_loss = loss.item() * yi.size(0)
    trainset_loss += batch_loss
  return trainset_loss



optimizer = torch.optim.Adam(model.parameters(), lr=lr)

trainset_loss1 = precision(prueba=True)

perdida1 = perdida(prueba=True)