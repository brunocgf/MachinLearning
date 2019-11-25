
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
# Red neuronal de 1 capa escondida con activaciones sigmoide
# ==================

class NNh1_sigm(nn.Module):
  def __init__(self, input_size, num_classes):

    super(NNh1_sigm, self).__init__()

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
# Red neuronal de 3 capa escondida con activaciones sigmoide
# ==================

class NNh3_sigm(nn.Module):
  def __init__(self, input_size, num_classes):

    super(NNh3_sigm, self).__init__()

    hidden_size1 = 500
    hidden_size2 = 100
    hidden_size3 = 30

    self.fc1 = nn.Linear(input_size, hidden_size1)
    self.fc2 = nn.Linear(hidden_size1, hidden_size2)
    self.fc3 = nn.Linear(hidden_size2, hidden_size3)
    self.fc4 = nn.Linear(hidden_size3, num_classes)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    out = self.sigmoid(self.fc1(x))
    out = self.sigmoid(self.fc2(out))
    out = self.sigmoid(self.fc3(out))
    out = self.sigmoid(self.fc4(out))
    return out


# ==================
# Red neuronal de 5 capa escondida con activaciones sigmoide
# ==================

class NNh5_sigm(nn.Module):
  def __init__(self, input_size, num_classes):

    super(NNh5_sigm, self).__init__()

    hidden_size1 = 500
    hidden_size2 = 300
    hidden_size3 = 100
    hidden_size4 = 50
    hidden_size5 = 30

    self.fc1 = nn.Linear(input_size, hidden_size1)
    self.fc2 = nn.Linear(hidden_size1, hidden_size2)
    self.fc3 = nn.Linear(hidden_size2, hidden_size3)
    self.fc4 = nn.Linear(hidden_size3, hidden_size4)
    self.fc5 = nn.Linear(hidden_size4, hidden_size5)
    self.fc6 = nn.Linear(hidden_size5, num_classes)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    out = self.sigmoid(self.fc1(x))
    out = self.sigmoid(self.fc2(out))
    out = self.sigmoid(self.fc3(out))
    out = self.sigmoid(self.fc4(out))
    out = self.sigmoid(self.fc5(out))
    out = self.sigmoid(self.fc6(out))
    return out

# Funciones RELU


class NNh1_relu(nn.Module):
  def __init__(self, input_size, num_classes):

    super(NNh1_relu, self).__init__()

    # [COMENTARIO]
    hidden_size = 500

    # [COMENTARIO]
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    return out



# ==================
# Red neuronal de 3 capa escondida con activaciones sigmoide
# ==================

class NNh3_relu(nn.Module):
  def __init__(self, input_size, num_classes):

    super(NNh3_relu, self).__init__()

    hidden_size1 = 500
    hidden_size2 = 100
    hidden_size3 = 30

    self.fc1 = nn.Linear(input_size, hidden_size1)
    self.fc2 = nn.Linear(hidden_size1, hidden_size2)
    self.fc3 = nn.Linear(hidden_size2, hidden_size3)
    self.fc4 = nn.Linear(hidden_size3, num_classes)
    self.relu = nn.ReLU()

  def forward(self, x):
    out = self.relu(self.fc1(x))
    out = self.relu(self.fc2(out))
    out = self.relu(self.fc3(out))
    out = self.relu(self.fc4(out))
    return out


# ==================
# Red neuronal de 5 capa escondida con activaciones sigmoide
# ==================

class NNh5_relu(nn.Module):
  def __init__(self, input_size, num_classes):

    super(NNh5_relu, self).__init__()

    hidden_size1 = 500
    hidden_size2 = 300
    hidden_size3 = 100
    hidden_size4 = 50
    hidden_size5 = 30

    self.fc1 = nn.Linear(input_size, hidden_size1)
    self.fc2 = nn.Linear(hidden_size1, hidden_size2)
    self.fc3 = nn.Linear(hidden_size2, hidden_size3)
    self.fc4 = nn.Linear(hidden_size3, hidden_size4)
    self.fc5 = nn.Linear(hidden_size4, hidden_size5)
    self.fc6 = nn.Linear(hidden_size5, num_classes)
    self.relu = nn.ReLU()

  def forward(self, x):
    out = self.relu(self.fc1(x))
    out = self.relu(self.fc2(out))
    out = self.relu(self.fc3(out))
    out = self.relu(self.fc4(out))
    out = self.relu(self.fc5(out))
    out = self.relu(self.fc6(out))
    return out

class NNh2_relu(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NNh2_relu, self).__init__()

        hidden_size1, hidden_size2  = 200, 60

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        return out
