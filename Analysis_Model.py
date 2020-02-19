import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

Model_Name="model-1582108192"
def graph(model_name):
    content=open("model.log","r").read().split("\n")

    times=[]
    accuracy=[]
    losses=[]
    val_accuracy=[]
    val_losses=[]
    i=0
    for line in content:
        if model_name in line:
            name,timestamp,acc,loss,val_acc,val_loss=line.split(",")

            times.append(i)
            i+=1
            accuracy.append(float(acc))
            losses.append((float(loss)))
            val_accuracy.append((float(val_acc)))
            val_losses.append((float(val_loss)))

    fig=plt.figure()

    ax1=plt.subplot2grid((2,1),(0,0))
    ax2=plt.subplot2grid((2,1),(1,0),sharex=ax1)

    ax1.plot(times,accuracy,label="Accuracy")
    ax1.plot(times,val_accuracy,label="Validation Acc")
    ax1.legend(loc=2)

    ax2.plot(times,losses,label="Loss")
    ax2.plot(times,val_losses,label="Validation loss")
    ax2.legend(loc=2)

    plt.show()

graph(Model_Name)

