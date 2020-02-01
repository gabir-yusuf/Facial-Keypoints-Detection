from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data_load import FacialKeypointsDataset
from data_load import Rescale, RandomCrop, Normalize, ToTensor
import torch.optim as optim

from models import *

# instantiate the model
net = AlexNet()
print(net)

# define the data transform using transfroms.Compose([..])
# Note the order is matter
data_tranform = transforms.Compose([Rescale((250, 250)),
                                    RandomCrop((227, 227)),
                                    Normalize(),
                                    ToTensor()])

# Create the transformed dataset
transformed_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',
                                             root_dir='data/training/',
                                             transform=data_tranform)

print('Number of images: ', len(transformed_dataset))

# iterate through the transformed dataset and print some stats about the first few samples
for i in range(4):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['keypoints'].size())

# Loadiing the data in batches
# for windows users, change the num_workers to 0 or you will face some issues with your DataLoader failing
batch_size = 32

train_loader = DataLoader(transformed_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4)

criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)


def train_net(n_epochs):
    # Model in training mode, dropout is on
    net.train()

    for epoch in range(n_epochs):
        running_loss = 0.0
        # train on batches of data
        for batch_i, data in enumerate(train_loader):
            # get the input images and theri crosponding labels
            images, key_pts = data['image'], data['keypoints']
            # Flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)
            # Wrap them in a torch Variable
            images, key_pts = Variable(images), Variable(key_pts)
            # convert the variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            # Forward pass to get the output
            output_pts = net.forward(images)
            # Calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)
            # zero the parameter (wight) gradients
            optimizer.zero_grad()
            # backword pass to calculate the weight gradients
            loss.backward()
            # update the weights
            optimizer.step()

            # print the loss statistics
            running_loss += loss.item()
            if batch_i % 32 == 31: # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i + 1, running_loss / 32))
                running_loss = 0.0

    print('Training Finished')

n_epochs = 25
train_net(n_epochs)

# saving the model parameters
model_dir = 'saved_models/'
model_name = 'keypoints_model_AlexNet_50epochs.pth'
torch.save(net.state_dict(), model_dir + model_name)
