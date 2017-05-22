import numpy as np
from numpy import linalg as LA

V = 30000 # size of the vocabulary
N = 300 # number of nodes in the hidden layer
ws = 2 # window size
C = 2*ws 

def main():
	x = np.random.rand(V,1) # input word represented as one-hot
	w = np.random.rand(N,V) # weights between input and hidden layer
	v = transpose(W) # weights between hidden layer and the output
	t = np.zeros((V,C)) # expected output word as one-hot 
	y = np.zeros((V,C)) # real output layer 

	h = np.dot(w,x)
	u = np.dot(transpose(w,h))
	y = np.exp(u - np.max(u))
	y = y / y.sum(axis=0)

	dv = np.multiply((y-t), h)
