import numpy as np

def sigmoid(x):
  #Calculates the sigmoid of the input x.
  return 1 / (1 + np.exp(-x))

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
  neural_network.append([w, b, activation])
  a_prev = activation #Updating the 'previous' activation for the next layer

for layer in neural_network:
  print(layer)

"""
Update: We conducted a FORWARD PASS to initialize activations in each layer

STAGE 2: We will now simulate BACK PROPAGATION
"""

