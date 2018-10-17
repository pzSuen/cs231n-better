from cs231n.classifiers import KNearestNeighbor
import numpy as np

classifier = KNearestNeighbor()
classifier.__init__()
print("I have import.")

a = np.arange(2000).reshape(100, 20)
a_y = np.array([1] * 5)
b = np.arange(20).reshape(1, 20)
c = a - b
print(a_y)
classifier.train(a, a_y.T)
classifier.compute_distances_one_loop(a)
