
# coding: utf-8
import time
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def knn(k, unknown, labels, positions):
    nearest_k = np.array([['unknown', 1<<15, [0]*len(positions[0])]] * k, dtype=object)
    for i, point in enumerate(positions):
        r = distance(unknown, point)
        if (r < nearest_k[-1][1]):
            nearest_k[-1] = np.array([labels[i], r, point], dtype=object)
            nearest_k = nearest_k[nearest_k[:,1].argsort()]
    return most_common(nearest_k), nearest_k

def distance(unknown_vector, labeled_vector):
    total = 0
    for uv, lv in zip(unknown_vector, labeled_vector):
        total += math.pow(lv-uv, 2)
    return math.sqrt(total)

def most_common(nearest_k):
    nearest_k_dict = dict()
    for label in nearest_k[:,0]:
        if(label in nearest_k_dict):
            nearest_k_dict[label] += 1
        else:
            nearest_k_dict[label] = 1
    nearest_neighbor = max(nearest_k_dict, key=nearest_k_dict.get)
    return nearest_neighbor

def main():
	np.random.seed(int(time.time()))
	sample_size = 30
	k=5
	data_1 = [["Blue", x, y, z] for x, y, z in zip(np.random.rand(sample_size), np.random.rand(sample_size), np.random.rand(sample_size))]
	data_2 = [["Orange", x, y, z] for x, y, z in zip(np.random.rand(sample_size), np.random.rand(sample_size), np.random.rand(sample_size))]
	data = np.array(data_1 + data_2)
	un_x, un_y, un_z = np.random.rand(), np.random.rand(), np.random.rand()
	labels = data[:,0]
	positions = np.array(data[:,1:], dtype=float)
	nn, nearest_k = knn(k, [un_x, un_y, un_z], labels, positions)
	x_1 = [row[1] for row in data_1]
	y_1 = [row[2] for row in data_1]
	z_1 = [row[3] for row in data_1]
	x_2 = [row[1] for row in data_2]
	y_2 = [row[2] for row in data_2]
	z_2 = [row[3] for row in data_2]
	x_k = [row[2][0] for row in nearest_k]
	y_k = [row[2][1] for row in nearest_k]
	z_k = [row[2][2] for row in nearest_k]

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x_1, y_1, z_1, s=100, marker='^')
	ax.scatter(x_2, y_2, z_2, s=100, marker='o')
	ax.scatter(x_k, y_k, z_k, s=150, marker='x')
	ax.scatter(un_x, un_y, un_z, s=100, marker='*')
	plt.title(nn)
	plt.show()

if __name__ == '__main__':
	main()

