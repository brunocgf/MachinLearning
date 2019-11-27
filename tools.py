import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

root ='./datos'# Carpeta donde se guardaran los datos
train_data = MNIST(root, train=True, transform=ToTensor(), download=True)
test_data  = MNIST(root, train=False, transform=ToTensor())

train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=bs, shuffle=False)

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
