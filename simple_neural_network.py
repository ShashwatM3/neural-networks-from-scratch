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
Step 2: Compute gradient of loss w.r.t each parameter (bias and weight)
"""

y = 0.0

def d_a_z(neuron):
  return sigmoid_derivative(neuron[3])

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

bias_b1 = (
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

bias_b2 = (
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

bias_b3 = (
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

bias_b4 = (
  d_a_z(neural_network[3]) *
  d_L_a(neural_network[3])
)

derivatives_gradients=[gradient_w1, gradient_w2, gradient_w3, gradient_w4]
derivatives_biases=[bias_b1, bias_b2, bias_b3, bias_b4]

"""
Update: We have successfully conducted 1 backpropagate round

Now we UPDATE the Weights + Biases of NN via GRADIENT DESCENT
---------------------------------------------------------------
1. Set the learning rate
2. use the general formula: [ param_new = param - learning_rate * derivative ]
"""

learning_rate = 0.01

for i in range(len(neural_network)):
  neural_network[i][0] = neural_network[i][0] - learning_rate*(derivatives_gradients[i])
  neural_network[i][1] = neural_network[i][1] - learning_rate*(derivatives_biases[i])