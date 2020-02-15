import os
import cv2
import numpy as np
from tqdm import tqdm



REBUILD_DATA = True

class Animal():
    IMG_SIZE=100
    CATS="Data/PetImages/Cat"
    DOGS="Data/PetImages/Dog"
    LABELS={CATS:0,DOGS:1}
    training_data=[]
    cat_count =0
    dog_count=0
    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                if "jpg" in f:
                    try :
                        path =os.path.join (label,f)
                        img =cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                        img=cv2.resize(img,(self.IMG_SIZE,self.IMG_SIZE))
                        self.training_data.append([np.array(img),np.eye(2)[self.LABELS[label]]])
                        # do something like print(np.eye(2)[1]), just makes one_hot vector

                        if label ==self.CATS:
                            self.cat_count+=1
                        elif label == self.DOGS:
                            self.dog_count+=1
                    except Exception as e:
                        print(e)
                        pass

        np.random.shuffle(self.training_data)
        np.save("training_data.npy",self.training_data)
        print("Cats:",self.cat_count)
        print("Dogs:",self.dog_count)
###
    if REBUILD_DATA:
        animal =Animal()
        animal.make_training_data()

    training_data=np.load("training_data.npy",allow_pickle=True)
    print(len(training_data))

###
import torch
import torch.nn as nn
import torch.nn.functional as Func
class Net(nn.module)
    def __init__(self):
        super().__init()
        self.conv1=nn.Conv2d(1,32,5)# input is 1 image, looking for 32 features , 5x5 window size
        #conv2d since image is in 2-dimension
        self.conv2=nn.Conv2d(32,64,5)
        self.conv3=nn.Conv2d(64,128,5)


