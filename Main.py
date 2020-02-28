import torch.optim as optim
from torchvision.models import mobilenet_v2
from Net import *
import time
from GuangFu import *


Img_size =100
Batch_Size = 50
Epochs = 10
Total_Building_GuangFu=5

BUILD_DATA =False
Rebuild_Data=False
guangfu = GuangFu(Img_size,Total_Building_GuangFu)
if BUILD_DATA:
    guangfu.make_training_data(Trained=False)
elif Rebuild_Data:
    guangfu.make_training_data(Trained=True)
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

img = torch.Tensor([i[0] for i in training_data]).view(-1, Img_size, Img_size,3)
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


Training=True
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
name=[]
#Test image that havent been seen by model before
for i in range (len(list(guangfu.LABELS.keys()))):
    txt=list(guangfu.LABELS.keys())[i]
    # txt=txt.split("/")
    #name.append(txt[len(txt)-1])
    name.append(txt)


Actual_lbl={"YunPing": 0, "StudentCenter": 1, "River": 2, "XiaoXiMen": 3, "Tree": 4}
def output_net(dir_path):
    total_test = 0
    correct = 0
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
            predicted_lbl=int(torch.argmax(output))
            if list(Actual_lbl.keys())[predicted_lbl] in path:
                print("Correct")
                correct=correct+1
            else :
                print(f"Incorrect,{list(Actual_lbl.keys())[predicted_lbl]}")
            total_test+=1

    print(f"Unseen Data Accuracy {round(correct/total_test,3)}")

Path= "Data/Test"
output_net(Path)


##
testingImage=torch.randn(Img_size,Img_size).view(-1,1,Img_size,Img_size).to(Device);

net=torch.jit.trace(net,testingImage)
net.save("mobilenet.pt")



##

import matplotlib.pyplot as plt
path="Data/Test/River.jpg"
img=cv2.imread(path,cv2.IMREAD_COLOR)
img=cv2.resize(img,(Img_size,Img_size))
print(img.shape)
Image=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
Image=cv2.resize(Image,(Img_size,Img_size))
print(Image.shape)
plt.imshow(training_data[1][0])
plt.show()
##

##
print(list(guangfu.LABELS)[0])


##

