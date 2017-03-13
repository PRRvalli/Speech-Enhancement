import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt
from os import listdir
from os.path import isdir, join

from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge
from keras.datasets import imdb
from keras.optimizers import SGD
import theano

from sklearn.decomposition import PCA,RandomizedPCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import itertools
import matplotlib.pyplot as plt


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

noise_level=raw_input('[0dB,5dB,10dB,15dB]')
noise=raw_input(['restaurant', 'station', 'car', 'exhibition', 'train', 'babble', 'airport', 'street'])



file=open(noise_level+'/'+noise+'/'+'speaker_list.txt','rb')
speaker=file.readlines()
file.close()

#file=listdir(noise_level+'dB/'+noise+'/')
a=[0,1,5,6,10,11,15,16,20,21,25,26]
test_file=[speaker[x] for x in a]
train_file=[speaker[x] for x in xrange(30) if x not in a]
test_x=np.array([-1,-1,-1])
train_x=np.array([-1,-1,-1])
test_y=np.array([])
train_y=np.array([])

test_size=np.zeros((len(test_file)+1,))	
count=1
n=1
for x in test_file:
    out=np.array([])
    temp=wavfile.read(noise_level+'/'+noise+'/'+x[0:-1])
    temp2=wavfile.read('clean/'+x[0:4]+'.wav')
    n=max(np.max(temp[1]),-1*np.min(temp[1]),np.max(temp2[1]),-1*np.min(temp2[1]))
    out=matrix_con((temp[1]*1.0)/n)
    #print out.shape
    test_x=np.vstack((test_x,out))
    #n=max(np.max(temp2[1]),-1*np.min(temp2[1]))
    test_y=np.hstack((test_y,(temp2[1]*1.0)/n))       
		

'''
print np.max(test_y)
print np.min(test_y)
print np.max(test_x)
print np.min(test_x)
'''
wavfile.write('test.wav', temp[0], test_y)
for x in train_file:
    out=np.array([])
    temp=wavfile.read(noise_level+'/'+noise+'/'+x[0:-1])
    temp2=wavfile.read('clean/'+x[0:4]+'.wav')
    n=max(np.max(temp[1]),-1*np.min(temp[1]),np.max(temp2[1]),-1*np.min(temp2[1]))
    out=matrix_con((temp[1]*1.0)/n)
    #print out.shape
    train_x=np.vstack((train_x,out))
    #n=max(np.max(temp2[1]),-1*np.min(temp2[1]))
    train_y=np.hstack((train_y,(temp2[1]*1.0)/n))      

'''print np.max(train_y)
print np.min(train_y)
print np.max(train_x)
print np.min(train_x)'''
wavfile.write('train.wav', temp[0], train_y)

train_x=train_x[1:len(train_x),:]
test_x=test_x[1:len(test_x),:]
print train_x.shape
print train_y.shape
print test_x.shape
print test_y.shape

np.random.seed(1337)

epochs = 400

sequence = Input(shape=(train_x.shape[1],), dtype='int32')  
#embedded = Embedding(max_features, 128, input_length=156)(sequence)
embedded = Embedding(train_x.shape[0], 64, input_length=train_x.shape[1])(sequence)

forwards = LSTM(64)(embedded)
backwards = LSTM(64, go_backwards=True)(embedded)

merged = merge([forwards, backwards], mode='sum', concat_axis=-1)
after_dp = Dropout(0.2)(merged)
output = Dense(1, activation='tanh')(after_dp)

model = Model(input=sequence, output=output)

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
print('Train...')
#model.fit(X_train, Y_train,batch_size=32,nb_epoch=100,validation_data=[X_test, Y_test])
model.fit(train_x, train_y, validation_split = 0.10, nb_epoch=epochs, batch_size=32)

pred_y=model.predict(test_x)

wavfile.write('pred.wav', temp[0], pred_y)


		


