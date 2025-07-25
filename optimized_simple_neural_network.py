import numpy as np

# -------------------- Setup --------------------

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    if x>0:
        return 1
    else:
        return 0

def loss_fn(y_pred, y_true):
    return (y_pred - y_true)**2

def d_loss(y_pred, y_true):
    return 2 * (y_pred - y_true)

# -------------------- Class Definition --------------------
"""
We know that the following is specific to each "Neuron":
> weight
> bias
> z = weight*inp + bias
> a = sigmoid(z)
> da(L)_dz(L) = sigmoid_derivative(a(L))
> dz(L)_da(L-1) = w(L)
> dz(L)_dw(L) = a(L-1)
"""

class NeuronLayer:
  def __init__(self, weight, bias, activation_func, activation_derivative):
    self.weight = weight
    self.bias = bias
    self.activation_func = activation_func
    self.activation_derivative = activation_derivative
    self.z = None
    self.a = None
    self.da_dz = None
    self.dz_da = self.weight
    self.a_prev = None
    self.dz_dw = None
  
  def forward(self, a_prev): # This is where the ACTIVATION and Z VALUES will be INITIALIZED through calculations
    self.a_prev = a_prev
    self.z = self.weight*a_prev + self.bias
    self.dz_dw = a_prev
    self.a = self.activation_func(self.z)
    return self.a
  
  def backward(self): # This is where the required derivatives are set and RETURNED
    self.dz_da = self.weight
    self.da_dz = self.activation_derivative(self.z) # Derivative takes Z not a
  
# -------------------- Pre-defined variables --------------------

weights = np.random.randn(4) * np.sqrt(2/1)
biases = np.zeros(4)
learning_rate = 0.01
layers = [] # NN
output = 0 # Placeholder value

for w, b in zip(weights, biases):
   layers.append(NeuronLayer(weight=w, bias=b, activation_func=relu, activation_derivative=relu_derivative))

# -------------------- Forward Pass + Backpropagation --------------------

def forwardPass(input_network):
  input_next = input_network
  for layer in layers:
    input_next = layer.forward(input_next)
  return input_next

def backwardPropagate(predicted, target):
  for layer in layers:
    layer.backward()

  derivatives_weights = []
  derivatives_biases = []

  loss_grad = d_loss(predicted, target)

  for i in range(len(layers)):
    derivative_chain = 1 # Placeholder value
    for j in reversed(range(i+1, len(layers))):
      derivative_chain *= layers[j].da_dz * layers[j].dz_da
    derivative_chain *= layers[i].da_dz
    derivatives_weights.append(derivative_chain * loss_grad * layers[i].dz_dw)
    derivatives_biases.append(derivative_chain * loss_grad)

  for ind in range(len(derivatives_weights)):
    layers[ind].weight = layers[ind].weight - learning_rate*derivatives_weights[ind]
    layers[ind].bias = layers[ind].bias - learning_rate*derivatives_biases[ind]

# -------------------- Running training (w/ epochs) --------------------

epochs = int(input("Enter number of epochs - "))

samples = [ # y = 2x + 1
    [-2.0, -3.0],
    [-1.5, -2.0],
    [-1.0, -1.0],
    [-0.5, 0.0],
    [0.0, 1.0],
    [0.2, 1.4],
    [0.4, 1.8],
    [0.6, 2.2],
    [0.8, 2.6],
    [1.0, 3.0],
    [1.2, 3.4],
    [1.4, 3.8],
    [1.6, 4.2],
    [1.8, 4.6],
    [2.0, 5.0],
] # Samples we will be training our network on

for epoch in range(epochs):
  total_loss = 0
  for sample in samples:
    inp = sample[0]
    tar = sample[1]
    prediction = forwardPass(inp)
    loss = loss_fn(prediction, tar)
    backwardPropagate(prediction, tar)
    total_loss += loss
  print(f"Epoch {epoch+1}: Avg Loss = {total_loss / len(samples):.4f}")