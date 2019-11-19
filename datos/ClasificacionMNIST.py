import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from modelos import RegresionMultinomial

# Se verifica si esta disponible CUDA,en caso contraro se usa el procesador
device = torch.device('cuda'if torch.cuda.is_available() else'cpu')

# Hiper-parametros
input_size = 784 # Dimension de datos de entrada (28 x 28)
num_classes = 10 # MNIST tiene 10 clases (numeros del 1 al 10)
num_epochs = 5 # Numero de epocas para entrenar
bs = 100 # Tamano de lote (batch_size)
lr = 0.001 # Tasa de aprendizaje

# Se bajan los datos MINST

root ='./datos'# Carpeta donde se guardaran los datos
train_data = MNIST(root, train=True, transform=ToTensor(), download=True)
test_data  = MNIST(root, train=False, transform=ToTensor())

# Se crea el iterable sobre el dataset
train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=bs, shuffle=False)

# ==================
# Definimos modelo
# ==================

# Se define el modelo 
model = RegresionMultinomial(input_size, num_classes).to(device)
loss_function = nn.CrossEntropyLoss()

# ==================
# Optimizacion
# ==================

# Implementa el gradiente descendiente para la optimizacion
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Entrenamiento del modelo
for epoch in range(num_epochs):
  for i, (xi, yi) in enumerate(train_loader):
    
    # Las entradas de la imagen se convierten en vectores
    xi = xi.reshape(-1, 28*28).to(device)# imagenes
    yi = yi.to(device)# etiquetas
    
    # Propagacion para adelante
    output = model(xi)
    loss = loss_function(output, yi)
    # Propagcion para atras y paso de optimizacion
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (i+1) % 100 == 0:
      print ('Epoca: {}/{}, Paso: {}/{}, Perdida: {:.5f}'.format(epoch+1,num_epochs, i+1, len(train_loader), loss.item()))
      
# Prueba del modelo
# Al probar, usamos torch.no_grad() porque reduce el uso de memoria y velocidad al evaluar
with torch.no_grad():
  correct = 0
  total = 0
  for xi, yi in test_loader:
    # Las entradas de la imagen se convierten en vectores
    xi = xi.reshape(-1, 28*28).to(device)
    yi = yi.to(device)
    
    # Se hacen las predicciones
    output = model(xi)
    _, predicted = torch.max(output.data, 1)
    
    total += yi.size(0)
    correct += (predicted == yi).sum().item()
    
  print(f'Precision del modelo en {total} imagenes: {100 * correct / total}')
  
  # Precision del modelo en 10000 imagenes: 83.31
  
 # [COMENTARIO]
 save_model = False
 if save_model is True:
   torch.save(model.state_dict(),'modelo.ckpt')
   
   
   
# ==================
# Definimos modelo RedNeuronal
# ==================

from modelos import RedNeuronal


# Se define el modelo 
model = RedNeuronal(input_size, num_classes).to(device)
loss_function = nn.CrossEntropyLoss()

# ==================
# Optimizacion
# ==================

# [COMENTARIO]
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Entrenamiento del modelo
for epoch in range(num_epochs):
  for i, (xi, yi) in enumerate(train_loader):
    
    # [COMENTARIO]
    xi = xi.reshape(-1, 28*28).to(device)# imagenes
    yi = yi.to(device)# etiquetas
    
    # Propagacion para adelante
    output = model(xi)
    loss = loss_function(output, yi)
    # Propagcion para atras y paso de optimizacion
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (i+1) % 100 == 0:
      print ('Epoca: {}/{}, Paso: {}/{}, Perdida: {:.5f}'.format(epoch+1,num_epochs, i+1, len(train_loader), loss.item()))
      
# Prueba del modelo
# Al probar, usamos torch.no_grad() porque [COMENTARIO]
with torch.no_grad():
  correct = 0
  total = 0
  for xi, yi in test_loader:
    # [COMENTARIO]
    xi = xi.reshape(-1, 28*28).to(device)
    yi = yi.to(device)
    
    # [COMENTARIO]
    output = model(xi)
    _, predicted = torch.max(output.data, 1)
    
    total += yi.size(0)
    correct += (predicted == yi).sum().item()
    
  print(f'Precision del modelo en {total} imagenes: {100 * correct / total}')
  
  # Precision del modelo en 10000 imagenes: 83.31
  
 # [COMENTARIO]
 save_model = False
 if save_model is True:
   torch.save(model.state_dict(),'modelo.ckpt')
   
   
   
   
# ==================
# Definimos modelo RedNeuronal3
# ==================

from modelos import RedNeuronal3


# Se define el modelo 
model = RedNeuronal3(input_size, num_classes).to(device)
loss_function = nn.CrossEntropyLoss()

# ==================
# Optimizacion
# ==================

# [COMENTARIO]
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Entrenamiento del modelo
for epoch in range(num_epochs):
  for i, (xi, yi) in enumerate(train_loader):
    
    # [COMENTARIO]
    xi = xi.reshape(-1, 28*28).to(device)# imagenes
    yi = yi.to(device)# etiquetas
    
    # Propagacion para adelante
    output = model(xi)
    loss = loss_function(output, yi)
    # Propagcion para atras y paso de optimizacion
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (i+1) % 100 == 0:
      print ('Epoca: {}/{}, Paso: {}/{}, Perdida: {:.5f}'.format(epoch+1,num_epochs, i+1, len(train_loader), loss.item()))
      
# Prueba del modelo
# Al probar, usamos torch.no_grad() porque [COMENTARIO]
with torch.no_grad():
  correct = 0
  total = 0
  for xi, yi in test_loader:
    # [COMENTARIO]
    xi = xi.reshape(-1, 28*28).to(device)
    yi = yi.to(device)
    
    # [COMENTARIO]
    output = model(xi)
    _, predicted = torch.max(output.data, 1)
    
    total += yi.size(0)
    correct += (predicted == yi).sum().item()
    
  print(f'Precision del modelo en {total} imagenes: {100 * correct / total}')
  
  # Precision del modelo en 10000 imagenes: 83.31
  
 # [COMENTARIO]
 save_model = False
 if save_model is True:
   torch.save(model.state_dict(),'modelo.ckpt')