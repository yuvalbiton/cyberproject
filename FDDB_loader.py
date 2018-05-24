import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

manager_file = 'manager.txt'
'''
def close_event():
    plt.close()


def pics_to_np(manager_file):
    with open(manager_file, 'rb') as f:
	pics = []
        data1 = f.read().split('\n')
	del data1[-1]
	for submanager in data1:
    	    data2= open(submanager,'rb')
	    data2 = data2.read()
	    data2 = data2.split('\n')
	    del data2[-1]
            for pic in data2:
		print pic
	        a=Image.open('data/'+pic+'.jpg')
	        b = np.array(a)
	        pics.append(b)
	return pics
'''
size = (500,500)
def load_dataset(manager_file = 'manager.txt'):
	array_manager = []
	data_manager = open(manager_file,'rb')
	data_manager = data_manager.read().split('\n')
	del data_manager[-1]
	for submanager in data_manager:
		data_submanager = open(submanager,'rb')
		data_submanager = data_submanager.read().split('\n')
		del data_submanager[-1]
		i=1
		while i < len(data_submanager):
			if i == 1:
				array_manager.append(data_submanager[(i-1):(int(data_submanager[i]) + i+1)])
				i += (int(data_submanager[i]) +2)
			elif i != 0 :
				array_manager.append(data_submanager[(i-1):(int(data_submanager[i]) + i+1)])
				i += (int(data_submanager[i]) +2)
	for face in range(len(array_manager)):
		img = Image.open('data/'+array_manager[face][0]+'.jpg')
		img = img.resize(size, Image.ANTIALIAS)
		img2 = np.array(img)
		try:
		    img2.reshape((3,500,500))
		    array_manager[face][0] = img2
		except:
		    array_manager[face][0] = 0
		    break
		
		img.close()
	return array_manager
		

load_dataset(manager_file)
