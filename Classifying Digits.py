"""
Now we will try to somehow classify digits
"""

import numpy as np
from scipy.special import softmax
from generate_images import ImagesSet

#---------------------------------------------------------------------------------------------------
# ---------------------------------------- Setup ---------------------------------------------------
#---------------------------------------------------------------------------------------------------

images = ImagesSet(iterations=7)
training_data = images.initialize_images()

learning_rate = 0.01

def relu(x):
  return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)
    
def softmax_func(z):
  output = softmax(z, axis=0)
  return output

def cross_entropy_loss(y_true, y_pred):
    # Small epsilon to avoid log(0)
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.sum(y_true * np.log(y_pred))

def one_hot(y, num_classes=10):
    vec = np.zeros(num_classes)
    vec[int(y)-1] = 1
    return vec

#---------------------------------------------------------------------------------------------------
# ------------------------------------ Forward Pass/Back propogate functions -----------------------
#---------------------------------------------------------------------------------------------------

w1 = np.random.randn(1600, 1600) * 0.1
b1 = np.zeros(1600)

w2 = np.random.randn(1600, 1600) * 0.1
b2 = np.zeros(1600)

w3 = np.random.randn(10, 1600) * 0.1
b3 = np.zeros(10)

network = []

def forwardPass(input):
  global z1, z2, z3, activation_1, activation_2, activation_3
  network = []

  # ------------------------------------ Input to H1 -------------------------------------------------

  z1 = w1 @ input + b1
  activation_1 = relu(z1)
  network.append([w1, b1, z1, activation_1])

  # ------------------------------------ H1 to H2 -------------------------------------------------

  z2 = w2 @ activation_1 + b2
  activation_2 = relu(z2)
  network.append([w2, b2, z2, activation_2])

  # ------------------------------------ H2 to OUTPUT ------------------------------------------------

  z3 = w3 @ activation_2 + b3
  activation_3 = softmax_func(z3)
  network.append([w3, b3, z3, activation_3])

  return activation_3, network

def backprop(y_true, prediction, inp):
  global w1, w2, w3, b1, b2, b3
  delta_3 = prediction - y_true
  dl_dw3 = np.outer(delta_3, activation_2)
  dl_db3 = delta_3

  delta_2 = (w3.T @ delta_3) * relu_derivative(z2)
  dl_dw2 = np.outer(delta_2, activation_1)
  dl_db2 = delta_2

  delta_1 = (w2.T @ delta_2) * relu_derivative(z1)
  dl_dw1 = np.outer(delta_1, inp)
  dl_db1 = delta_1

  w3 -= learning_rate * dl_dw3
  b3 -= learning_rate * dl_db3

  w2 -= learning_rate * dl_dw2
  b2 -= learning_rate * dl_db2

  w1 -= learning_rate * dl_dw1
  b1 -= learning_rate * dl_db1

  return

#---------------------------------------------------------------------------------------------------
# ------------------------------------------- Simulation -------------------------------------------
#---------------------------------------------------------------------------------------------------

epochs = int(input("Enter number of training epochs - "))

for epoch in range(epochs):
  print(f"--------epoch {epoch}---------")
  total_losses = 0
  total=0
  for sample in training_data:
    input_sample = np.array(sample[0]) / 255.0
    prediction, network = forwardPass(input_sample)
    y_true = one_hot(sample[1][0])
    loss = cross_entropy_loss(y_true, prediction)
    backprop(y_true, prediction, input_sample)
    predicted_class = np.argmax(prediction)
    total+=1
    total_losses += 1 if predicted_class!=np.argmax(y_true) else 0
  print(f"Average Loss ===> {total_losses/total}")
  print("-------------------------")
  np.random.shuffle(training_data)

#---------------------------------------------------------------------------------------------------
# ------------------------------------------- Testing ----------------------------------------------
#---------------------------------------------------------------------------------------------------

image_paths = [
   [["test/draw4.png"], [4]],
   [["test/draw7.png"], [7]],
   [["test/draw8.png"], [8]]
]

testing_data = images.convertImages(image_paths)
for sample in testing_data:
  input_sample = np.array(sample[0]) / 255.0
  prediction, network = forwardPass(input_sample)
  print(np.argmax(prediction))