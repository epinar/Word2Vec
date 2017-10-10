import numpy as np
from numpy import linalg as LA
import glob, os
import re
import math
from collections import Counter
from random import randint
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

##### READ THE DATA AND CONSTRUCT THE DICTIONARY #####

data = []
dicti = {}
inverse_dict = {}
banned_words = ['ise', 'in', 'de', 'da',"ta", 've', 'ile', 'icin', 'nin', 'nın', "ın", "un", "için", "ı", "i", "o"]


V = 50000 # size of the vocabulary
N = 300 # number of nodes in the hidden layer
ws = 2 # window size
# C = 2*ws
NK = 5

def words(text): return re.findall(r'\w+', text)

def reader(directory):
	os.chdir(directory)
	global data
	termIdx = 0;
	for file in glob.glob("*.txt"):
		corpus = words(open(file).read())
		data += corpus

	vocabularies = Counter(data).most_common(V)

	# print(corp)
	print('vocab size', len(vocabularies))
	for i, val in enumerate(vocabularies):
		if val in banned_words:
		 	continue
		dicti[i] = val[0]
        # if val[0] == 'iran':
            # print i, val
		inverse_dict[val[0]] = i

##### CONSTRUCT AND UPDATE THE NEURAL NETWORK #####


class NeuralNetwork(object):

	trainNumber = 0
	alpha = 0.08 # learning rate
	loss = 0

	def __init__(self):
		self.W = 0.1 * np.random.randn(V,N) # weights between input and hidden layer
		self.V = 0.1 * np.random.randn(N,V) # weights between hidden layer and the output

	def train(self, x_k, y_k):

		n_k = NeuralNetwork.random_generate()
		C = len(y_k)
		K = len(y_k) + len(n_k)

		NeuralNetwork.trainNumber += 1
		# print(NeuralNetwork.trainNumber)
		if NeuralNetwork.trainNumber % 10000 == 0:

			print('word number is: ',NeuralNetwork.trainNumber)
			#print('W matrix : ')
			#print(self.W)
			#print('\n\n\nV matrix')
			#print(self.V)
			#print('\n\n\n\n')


		w = np.zeros((K, N)) #-> self.W'den doldur satirlar
		for i in range(0, len(y_k)):
			w[i, :] = self.W[y_k[i-1], :]
		for i in range(len(y_k), K):
			w[i, :] = self.W[n_k[i-len(y_k)], :]

		v = np.zeros((N, K)) #-> bunu da self.V'den doldur sutunlar
		for i in range(0, len(y_k)):
			v[:, i] = self.V[:, y_k[i]]
		for i in range(len(y_k), K):
			v[:, i] = self.V[:, n_k[i-len(y_k)]]

		h = np.transpose(np.matrix(self.W[x_k, :]))

		u_c = np.dot(np.transpose(v),h)
		u = np.tile(np.transpose(u_c),(len(y_k),1))

		y = np.exp(u - u.max(0))
		y = y / y.sum(axis=0) 
		y = np.linalg.norm(y)

		scoreMatExp = np.exp(np.asarray(u))
		scoreMatExp = scoreMatExp / scoreMatExp.sum(0)
		y = np.matrix(scoreMatExp)
		y = np.nan_to_num(y)

		t = np.zeros((C, K))
		for i in range(0, len(y_k)):
			t[i, i] = 1

		diff = np.subtract(y,t)
		diff_sum = np.matrix(np.sum(diff, axis = 0))

		#NeuralNetwork.loss = NeuralNetwork.loss + np.sum(diff_sum)
		#if NeuralNetwork.trainNumber % 10000 == 0:
		#	print(u)
		#	print(y)
		#	print(t)
		#	print(np.sum(diff_sum))
		#	NeuralNetwork.loss = NeuralNetwork.loss/10000
		#	print ("Average loss= ", NeuralNetwork.loss )
		#	NeuralNetwork.loss = 0

		dv = np.dot(h, diff_sum)
		dw = np.dot(diff_sum, w)

		for i in range(0, len(y_k)):
			self.V[:, y_k[i]] = np.subtract(self.V[:, y_k[i]], np.squeeze(np.asarray((np.multiply(NeuralNetwork.alpha, dv[:,i])))))

		for i in range(len(y_k), K):
			self.V[:, n_k[i-len(y_k)]] = np.subtract(self.V[:, n_k[i-len(y_k)]], np.squeeze(np.asarray(np.multiply(NeuralNetwork.alpha, dv[:, i]))))

		self.W[x_k,:] = np.subtract(self.W[x_k,:], np.multiply(NeuralNetwork.alpha, dw))

	def random_generate():
		n_k = []
		for i in range(0, NK):
			n_k.append(randint(0, V-1))
		return n_k


##### plot #####
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  print('plot with labels')
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

def construct_w():
	nn = NeuralNetwork()
	# for i, val in enumerate(data):
	print('data size: ', len(data)/10)
	for i in range(ws+1, int(len(data)/10) - ws - 1):
		val = data[i]
		x_k = inverse_dict.get(val, -1)

		if x_k == -1: continue
		y = []
		for j in range(i-ws, i+ws+1):
			if i == j: continue
			ind = inverse_dict.get(data[j], -1)
			if ind != -1:
				y.append(ind)
			else:
				y.append(V-1)
		# print('x_k: ', x_k, 'y: ', y)
		if len(y) == 2 * ws:
			nn.train(x_k, y)

	tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)	
	plot_only = 150
	nn.W = np.nan_to_num(nn.W)
	print('son W:', nn.W)
	low_dim_embs = tsne.fit_transform(nn.W[:plot_only, :])
	labels = []
	
	for i in range(0, plot_only):
		labels.append(dicti[i])
		# labels = [dicti[i] for i in xrange(plot_only)]
	plot_with_labels(low_dim_embs, labels)

	# NNs.append(nn)

# NNs = []


##### MAIN #####

def main():
	reader("data/") # reads the data
	construct_w()



main()
