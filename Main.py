import os
import cv2
import numpy as np
import torch.optim as optim
from Net import *
from tqdm import tqdm
import time



Img_size =50
Batch_Size = 50
Epochs = 10
Total_Building_GuangFu=4

class GuangFu():
    Total_Building=Total_Building_GuangFu
    YunPing = "Data/Building/YunPing"
    StudentCenter = "Data/Building/StudentCenter"
    River= "Data/Building/River"
    XiaoXiMen= "Data/Building/XiaoXiMen"
    LABELS = {YunPing: 0, StudentCenter: 1,River:2,XiaoXiMen:3}
    training_data = []
    River_count= 0
    XiaoXiMen_count= 0
    def __init__(self,img_size):
        self.IMG_SIZE = img_size


    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    self.training_data.append([np.array(img), np.eye(self.Total_Building)[self.LABELS[label]]])
                    # do something like print(np.eye(2)[1]), just makes one_hot vector

                    if label == self.River:
                        self.River_count+= 1
                    elif label == self.XiaoXiMen:
                        self.XiaoXiMen_count+= 1
                except Exception as e:
                    print(e)
                    pass

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("River:", self.River_count)
        print("XiaoXiMen:", self.XiaoXiMen_count)


###
REBUILD_DATA =False
if REBUILD_DATA:
    guangfu= GuangFu(Img_size)
    guangfu.make_training_data()
training_data = np.load("training_data.npy", allow_pickle=True)
print(len(training_data))

###

if torch.cuda.is_available():
    print("Running in GPU")
    Device = torch.device("cuda:0")
    # Try to run it in GPU
else:
    Device = torch.device("cpu")
    print("Running in CPU")
net = Net(total_building=Total_Building_GuangFu,img_size=Img_size).to(Device)

###

img = torch.Tensor([i[0] for i in training_data]).view(-1, Img_size, Img_size)
img = img / 255.0  # Scaling the value of each pixel to 0~1
lbl = torch.Tensor([i[1] for i in training_data])

Test_Percentage = 0.1
Test_set = int(len(img) * Test_Percentage)

Val_Percentage = 0.1
Val_set = int(len(img) * Val_Percentage)

Train_img = img[:-(Test_set + Val_set)]
Train_lbl = lbl[:-(Test_set + Val_set)]

Test_img = img[-(Test_set + Val_set):-Val_set]
Test_lbl = lbl[-(Test_set + Val_set):-Val_set]

Val_img = img[-Val_set:]
Val_lbl = lbl[-Val_set:]
print(f"Training data count :{len(Train_img)}")
print(f"Testing data count :{len(Test_img)}")
print(f"Validation data count :{len(Val_img)}")

Optimizer = optim.Adam(net.parameters(), lr=0.001)
Loss_function = nn.MSELoss()

Model_Name = f"model-{int(time.time())}"

##
print(f"training_Data:{len(training_data)}")
print(f"Actul train data:{len(img)}")
print(f"Actual label:{len(lbl)}")
##
from Net import *
def fwd_pass(img, lbl, train=False):
    if train:
        net.zero_grad()
    output = net(img)
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(output, lbl)]
    acc = matches.count(True) / len(matches)
    loss = Loss_function(output, lbl)
    if train:
        loss.backward()
        Optimizer.step()
    return acc, loss

def train():
    with open("model.log", "w") as log_file:
        for epoch in range(Epochs):
            for i in tqdm(range(0, len(Train_img), Batch_Size)):
                batch_img = Train_img[i:i + Batch_Size].view(-1, 1, Img_size, Img_size).to(Device)
                batch_lbl = Train_lbl[i:i + Batch_Size].to(Device)

                acc, loss = fwd_pass(batch_img, batch_lbl, train=True)
                if i % 1 == 0:
                    val_acc, val_loss = batch_test(size=50)
                    log_file.write(f"{Model_Name},{int(time.time())},{round(float(acc), 3)},{round(float(loss), 3)}")
                    log_file.write(f",{round(float(val_acc), 3)},{round(float(val_loss), 3)}\n")
            print(f"Epoch:{epoch},Lose:{loss}")


def test(net, test=True):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(Val_img))):
            if test:
                real_lbl = torch.argmax(Test_lbl[i]).to(Device)
                net_out = net(Test_img[i].view(-1, 1, Img_size, Img_size).to(Device))[0]
             else:
                real_lbl = torch.argmax(Val_lbl[i]).to(Device)
                net_out = net(Val_img[i].view(-1, 1, Img_size, Img_size).to(Device))[0]

            predicted_lbl = torch.argmax(net_out)
            if predicted_lbl == real_lbl:
                correct += 1
            total += 1
    print("Accuracy:", round(correct / total, 3))


def batch_test(size=32):
    random_starting_pt = np.random.randint(len(Test_img) - size)
    img = Test_img[random_starting_pt:random_starting_pt + size]
    lbl = Test_lbl[random_starting_pt:random_starting_pt + size]
    with torch.no_grad():
        # not wasting time to calculate the gradient but we can if we want
        val_acc, val_loss = fwd_pass(img.view(-1, 1, Img_size, Img_size).to(Device), lbl.to(Device), train=False)
    return val_acc, val_loss


Training=False
if Training:
    train()
    print("Train Set :")
    test(net)
    print("Val_Set:")
    test(net,test=False)
    torch.save(net,f="Model_Trained.pt")
else :
    net=torch.load(f="Model_Trained.pt")
    net.eval()


##
test(net,test=False)
##
def output_net(dir_path):
    for f in os.listdir(dir_path):
        path=os.path.join(dir_path,f)
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(Img_size,Img_size))
        img=torch.Tensor(img).view(-1,Img_size,Img_size)
        img=img/255
        with torch.no_grad():
            output=net(img.view(-1,1,Img_size,Img_size).to(Device))
            print(f)
            print(torch.argmax(output))
Path= "Data/Test"
output_net(Path)


##

import matplotlib.pyplot as plt
plt.imshow(Test_img[1][0])
plt.show()
