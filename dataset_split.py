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

noise_level=raw_input('[0dB,5dB,10dB,15dB]')
noise=raw_input(['restaurant', 'station', 'car', 'exhibition', 'train', 'babble', 'airport', 'street'])



file=open(noise_level+'/'+noise+'/'+'speaker_list.txt','rb')
speaker=file.readlines()
file.close()

#file=listdir(noise_level+'dB/'+noise+'/')
a=[0,1,5,6,10,11,15,16,20,21,25,26]
test_file=[speaker[x] for x in a]
train_file=[speaker[x] for x in xrange(30) if x not in a]
test_x=np.array([])
train_x=np.array([])
test_y=np.array([])
train_y=np.array([])

test_size=np.zeros((len(test_file)+1,))	
count=1
for x in test_file:
        temp=wavfile.read(noise_level+'/'+noise+'/'+x[0:-1])
        temp2=wavfile.read('clean/'+x[0:4]+'.wav')
        test_y=np.hstack((test_y,10))
        test_y=np.hstack((test_y,(temp2[1]*1.0)/np.max(temp2[1])))

   
        #print len(temp[1])
        test_x=np.hstack((test_x,10))
        test_x=np.hstack((test_x,(temp[1]*1.0)/np.max(temp[1])))
        test_size[count]=(len(test_x)-1)
        count+=1
test_x=np.hstack((test_x,10))
test_y=np.hstack((test_y,10))
print len(test_x)
print np.sum(test_size)

	
train_size=np.zeros((len(train_file)+1,))	
count=1
for x in train_file:
        temp=wavfile.read(noise_level+'/'+noise+'/'+x[0:-1])
        temp2=wavfile.read('clean/'+x[0:4]+'.wav')
        train_y=np.hstack((train_y,10))
        train_y=np.hstack((train_y,(temp2[1]*1.0)/np.max(temp2[1])))

        #print len(temp[1])
        
        train_x=np.hstack((train_x,10))
        train_x=np.hstack((train_x,(temp[1]*1.0)/np.max(temp[1])))
        train_size[count]=(len(train_x)-1)
        count+=1
train_x=np.hstack((train_x,10)) 
train_y=np.hstack((train_y,10)) 


print len(train_x)
print np.sum(train_size)

Y_train=train_y-train_x
Y_test=test_y-test_x

for x in train_size:     
	print train_x[x]
for x in test_size:      
	print test_x[x]

'''#print test.shape
plt.plot(test_x,color='blue')
plt.plot(test_y,color='green')
plt.show()

plt.figure()

plt.plot(train_x,color='blue')
plt.plot(train_y,color='green')
plt.show()

plt.figure()

plt.plot(Y_train[train_size],color='blue')
plt.show()

plt.figure()
plt.plot(Y_test[test_size],color='green')
plt.show()
'''
#print type(train)
#print type(temp)

