import gensim.downloader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the GloVe model
model = gensim.downloader.load("glove-wiki-gigaword-50")

# Words to visualize
words = ["tower", "building", "skyscraper", "roof", "dome", "facade", 
         "lighthouse", "house", "bridge", "monument", "structure"]

# Get vectors for each word
vectors = np.array([model[word] for word in words])

# Reduce dimensionality to 2D using PCA
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# Create the plot
plt.figure(figsize=(10, 8))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], color='blue')

# Add labels for each point
for i, word in enumerate(words):
    plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]), 
                 xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

plt.title("Word Vector Visualization (PCA-reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()