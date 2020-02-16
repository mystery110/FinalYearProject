import os
import cv2
import numpy as np
from tqdm import tqdm

REBUILD_DATA = False


class Animal():
    IMG_SIZE = 100
    CATS = "Data/PetImages/Cat"
    DOGS = "Data/PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []
    cat_count = 0
    dog_count = 0

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

                        if label == self.CATS:
                            self.cat_count += 1
                        elif label == self.DOGS:
                            self.dog_count += 1
                    except Exception as e:
                        print(e)
                        pass

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("Cats:", self.cat_count)
        print("Dogs:", self.dog_count)


###
if REBUILD_DATA:
    animal = Animal()
    animal.make_training_data()
training_data = np.load("training_data.npy", allow_pickle=True)
print(len(training_data))

###
from Net import *

if torch.cuda.is_available():
    print("Running in GPU")
    Device = torch.device("cuda:0")
    # Try to run it in GPU
else:
    Device = torch.device("cpu")
    print("Running in CPU")
net = Net().to(Device)
###
import torch.optim as optim

img = torch.Tensor([i[0] for i in training_data]).view(-1, 100, 100)
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

##
Batch_Size = 100
Epochs = 2


def train(net):
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    for epoch in range(Epochs):
        for i in tqdm(range(0, len(Train_img), Batch_Size)):
            batch_img = Train_img[i:i + Batch_Size].view(-1, 1, 100, 100)
            batch_lbl = Train_lbl[i:i + Batch_Size]
            batch_img, batch_lbl = batch_img.to(Device),batch_lbl.to(Device)

            net.zero_grad()
            output = net(batch_img)
            loss = loss_function(output, batch_lbl)
            loss.backward()
            optimizer.step()
        print(f"Epoch:{epoch},Lose{loss}")


train(net)

##
def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(Test_img))):
            real_lbl = torch.argmax(Test_lbl[i])
            net_out = net(Test_img[i].view(-1, 1, 100, 100))[0]

            predicted_lbl = torch.argmax(net_out)
            if predicted_lbl == real_lbl:
                correct += 1
            total += 1
    print("Accuracy:", round(correct / total, 3))
