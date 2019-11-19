import torch.nn as nn
import torch.nn.functional as F
# ==================
# Regresion logistica multinomial
# ==================
class RegresionMultinomial(nn.Module):
  def __init__(self, input_size, num_classes):
    super(RegresionMultinomial, self).__init__()
    self.linear = nn.Linear(input_size, num_classes)
    
  def forward(self, x):
    out = self.linear(x)
    return out


# ==================
# Red neuronal de 1 capa escondida con activaciones ReLU
# ==================

class RedNeuronal(nn.Module):
  def __init__(self, input_size, num_classes):
    
    # [COMENTARIO]
    super(RedNeuronal, self).__init__()
    
    # [COMENTARIO]
    hidden_size = 500
    
    # [COMENTARIO]
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.sigmoid = nn.Sigmoid()
    self.fc2 = nn.Linear(hidden_size, num_classes)
    
  def forward(self, x):
    out = self.fc1(x)
    out = self.sigmoid(out)
    out = self.fc2(out)
    return out
  
  
  
# ==================
# Red neuronal de 3 capa escondida con activaciones ReLU
# ==================

class RedNeuronal3(nn.Module):
  def __init__(self, input_size, num_classes):
    
    # [COMENTARIO]
    super(RedNeuronal3, self).__init__()
    
    # [COMENTARIO]
    hidden_size1 = 500
    hidden_size2 = 100
    hidden_size3 = 30
    
    # [COMENTARIO]
    self.fc1 = nn.Linear(input_size, hidden_size1)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size1, hidden_size2)
    self.fc3 = nn.Linear(hidden_size2, hidden_size3)
    self.fc4 = nn.Linear(hidden_size3, num_classes)
    
  def forward(self, x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    out = self.fc3(out)
    out = self.fc4(out)
    return out
  