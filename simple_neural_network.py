import numpy as np

def sigmoid(x):
  #Calculates the sigmoid of the input x.
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sigmoid_x):
  return sigmoid_x * (1 - sigmoid_x)

#Even though the weights & biases exist randomly at the start, the activations (a) must be calculated forward as they don't exist until the forward pass is done.

np.random.seed(0)

#Initializing one weight and bias per layer
weights = [0.5, -0.4, 0.8, 0.3]
biases = [0.1, -0.2, 0.05, 0.0]

#Doing the forward pass
neural_network = []
x = 1.0
a_prev = x

for w, b in zip(weights, biases):
  z = w*a_prev + b
  activation = sigmoid(z)
  neural_network.append([w, b, z, activation])
  a_prev = activation #Updating the 'previous' activation for the next layer

for layer in neural_network:
  print(layer)

"""
Update: We conducted a FORWARD PASS to initialize activations in each layer

STAGE 2: We will now simulate BACK PROPAGATION
-----------------------------------------------
Overview of steps to be covered:
Step 1: We calculate the output & loss. For this example, let's just assume the output to be y = 0.0
Step 2: Compute gradient of loss w.r.t each parameter
"""

y = 0.0

def d_a_z(neuron):
  return sigmoid_derivative(neuron[2])

def d_L_a(neuron):
  return 2 * (neuron[3] - y)

def d_z_a_prev(neuron):
  return neuron[0]

# -----------------------------------------------------------------
### Layer 1 — Gradients
gradient_w1 = (
  d_a_z(neural_network[3]) *
  d_L_a(neural_network[3]) * 
  d_z_a_prev(neural_network[3]) *
  d_a_z(neural_network[2]) * 
  d_z_a_prev(neural_network[2]) *
  d_a_z(neural_network[1]) * 
  d_z_a_prev(neural_network[1]) *
  d_a_z(neural_network[0]) * 
  x
)

gradient_b1 = (
  d_a_z(neural_network[3]) *
  d_L_a(neural_network[3]) * 
  d_z_a_prev(neural_network[3]) *
  d_a_z(neural_network[2]) * 
  d_z_a_prev(neural_network[2]) *
  d_a_z(neural_network[1]) * 
  d_z_a_prev(neural_network[1]) *
  d_a_z(neural_network[0])
)
# -----------------------------------------------------------------
### Layer 2 — Gradients
gradient_w2 = (
  d_a_z(neural_network[3]) *
  d_L_a(neural_network[3]) * 
  d_z_a_prev(neural_network[3]) *
  d_a_z(neural_network[2]) * 
  d_z_a_prev(neural_network[2]) *
  d_a_z(neural_network[1]) * 
  neural_network[0][3]
)

gradient_b2 = (
  d_a_z(neural_network[3]) *
  d_L_a(neural_network[3]) * 
  d_z_a_prev(neural_network[3]) *
  d_a_z(neural_network[2]) * 
  d_z_a_prev(neural_network[2]) *
  d_a_z(neural_network[1])
)
# -----------------------------------------------------------------
### Layer 3 — Gradients
gradient_w3 = (
  d_a_z(neural_network[3]) *
  d_L_a(neural_network[3]) * 
  d_z_a_prev(neural_network[3]) *
  d_a_z(neural_network[2]) * 
  neural_network[1][3]
)

gradient_b3 = (
  d_a_z(neural_network[3]) *
  d_L_a(neural_network[3]) * 
  d_z_a_prev(neural_network[3]) *
  d_a_z(neural_network[2])
)

# -----------------------------------------------------------------
### Layer 4 — Gradients
gradient_w4 = (
  d_a_z(neural_network[3]) *
  d_L_a(neural_network[3]) * 
  neural_network[2][3]
)

gradient_b4 = (
  d_a_z(neural_network[3]) *
  d_L_a(neural_network[3])
)