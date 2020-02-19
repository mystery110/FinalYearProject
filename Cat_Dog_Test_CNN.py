import os
import cv2
import numpy as np
import torch.optim as optim
from Net import *
from tqdm import tqdm
import time

REBUILD_DATA =False

Img_size = 100
Batch_Size = 100
Epochs = 5


class Animal():
    IMG_SIZE = 100
    YunPing = "Data/Building/YunPing"
    StudentCenter = "Data/Building/StudentCenter"
    LABELS = {YunPing: 0, StudentCenter: 1}
    training_data = []
    YunPing = 0
    StudentCenter = 0

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                if "jpg" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])
                        # do something like print(np.eye(2)[1]), just makes one_hot vector

                        if label == self.YunPing:
                            self.YunPing += 1
                        elif label == self.StudentCenter:
                            self.StudentCenter += 1
                    except Exception as e:
                        print(e)
                        pass

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("YunPing:", self.YunPing)
        print("StudentCenter:", self.StudentCenter)


###
if REBUILD_DATA:
    animal = Animal()
    animal.make_training_data()
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
net = Net().to(Device)

###

img = torch.Tensor([i[0] for i in training_data]).view(-1, Img_size, Img_size)
img = img / 255.0  # Scaling the value of each pixel to 0~1
lbl = torch.Tensor([i[1] for i in training_data])

Val_Percentage = 0.1
Val_set = int(len(img) * Val_Percentage)

Train_img = img[:-Val_set]
Train_lbl = lbl[:-Val_set]

Test_img = img[-Val_set:]
Test_lbl = lbl[-Val_set:]
print(len(Train_img))
print(len(Test_img))

Optimizer = optim.Adam(net.parameters(), lr=0.001)
Loss_function = nn.MSELoss()

Model_Name = f"model-{int(time.time())}"


##
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
                if i % 20 == 0:
                    val_acc,val_loss=batch_test(size=50)
                    log_file.write(f"{Model_Name},{int(time.time())},{round(float(acc),3)},{round(float(loss),3)}")
                    log_file.write(f",{round(float(val_acc),3)},{round(float(val_loss),3)}\n")
            print(f"Epoch:{epoch},Lose:{loss}")


def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(Test_img))):
            real_lbl = torch.argmax(Test_lbl[i]).to(Device)
            net_out = net(Test_img[i].view(-1, 1, Img_size, Img_size).to(Device))[0]

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


train()
test(net)

##
