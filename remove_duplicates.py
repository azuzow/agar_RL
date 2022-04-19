import hashlib
# from scipy.misc import imread, imresize, imshow
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import numpy as np
import os
def file_hash(filepath):
    with open(filepath, 'rb') as f:
        return md5(f.read()).hexdigest()

for i in range(11):
	os.chdir(r'/home/alexzuzow/Desktop/agar_multiagent/DATA/Train/'+str(i))
	file_list = os.listdir()

	duplicates = []
	hash_keys = dict()
	for index, filename in  enumerate(os.listdir('.')):  #listdir('.') = current directory
	    if os.path.isfile(filename):
	        with open(filename, 'rb') as f:
	            filehash = hashlib.md5(f.read()).hexdigest()
	        if filehash not in hash_keys: 
	            hash_keys[filehash] = index
	        else:
	            duplicates.append((index,hash_keys[filehash]))

	for index in duplicates:
		os.remove(file_list[index[0]])