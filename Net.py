import torch
import torch.nn as nn
import torch.nn.functional as Func
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        # input is 1 image, looking for 32 features , 5x5 window size
        # conv2d since image is in 2-dimension
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        testing_image = torch.randn(100, 100).view(-1, 1, 100, 100)
        # create a random data to test what is the output of the vector when it is flatten
        self._to_linear = None
        self.passing_conv(testing_image)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def passing_conv(self, image):
        image = Func.relu(self.conv1(image))
        image = Func.max_pool2d(image, (2, 2))
        image = Func.relu(self.conv2(image))
        image = Func.max_pool2d(image, (2, 2))
        image = Func.relu(self.conv3(image))
        image = Func.max_pool2d(image, (2, 2))

        if self._to_linear is None:
            self._to_linear = image[0].shape[0] * image[0].shape[1] * image[0].shape[2]
        return image

    def forward(self, image):
        image = self.passing_conv(image)
        image = image.view(-1, self._to_linear)  # flatten the output
        image = Func.relu(self.fc1(image))
        image = self.fc2(image)
        return Func.softmax(image, dim=1)
