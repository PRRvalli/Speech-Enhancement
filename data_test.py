import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt
from os import listdir
from os.path import isdir, join


def matrix_con(arr):
	n=len(arr)
	arr=np.hstack((0,arr))
	arr=np.hstack((arr,0))
	#print arr
	i=1
	out=np.zeros((n,3))
	while i < (len(arr)-1):
		out[i-1,:]=arr[i-1:i+2]
		i+=1
	return out	
		
X=matrix_con([5,4,6,2])
Y=matrix_con([2,3,4,1])
out=np.vstack((X,Y))
print out 




