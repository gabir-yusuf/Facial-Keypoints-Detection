from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data_load import FacialKeypointsDataset
from data_load import Rescale, RandomCrop, Normalize, ToTensor
import torch.optim as optim

from models import *
import warnings

warnings.simplefilter("ignore")


def dataset(scale, crop_size, csv_file, root_dir):
    """
    apply the required processing on data before training
    Args:
        scale: (tuple or int) the size that the image will be resized to
        crop_size: (tuple or int) the input size to the model. square size is recommended
        csv_file: (str) directory to the csv file
        root_dir: (str) root directory for training data
    """
    # define the data transform. Note the order is matter
    data_transform = transforms.Compose([Rescale(scale),
                                         RandomCrop(crop_size),
                                         Normalize(),
                                         ToTensor()])

    # Create the transformed dataset
    transformed_dataset = FacialKeypointsDataset(csv_file, root_dir, data_transform)

    # print some stats about the first few samples
    print('Number of images: ', len(transformed_dataset))
    for i in range(4):
        sample = transformed_dataset[i]
        print(i, sample['image'].size(), sample['keypoints'].size())

    return transformed_dataset


def train(epochs_num, data_loader):
    """
    loop through dataset and train the model.
    print training status every 10 batches

    args:
    n_epochs: (int) number of epoch for training
    """
    # Model in training mode, dropout is on
    net.train()

    for epoch in range(epochs_num):
        running_loss = 0.0
        # train on batches of data
        for batch_i, data in enumerate(data_loader):
            # get the input images and their compounding labels
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
            # backward pass to calculate the weight gradients
            loss.backward()
            # update the weights
            optimizer.step()

            # print the loss statistics
            running_loss += loss.item()
            if batch_i % 10 == 9:  # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i + 1, running_loss / 10))
                running_loss = 0.0

    print('Training Finished')


if __name__ == "__main__":
    # instantiate the model
    net = AlexNet()
    print(net)

    transformed_dataset = dataset(scale=(250, 250),
                                  crop_size=(227, 227),
                                  csv_file='data/training_frames_keypoints.csv',
                                  root_dir='data/training/')

    # Loading the data in batches
    # for windows users, change the num_workers to 0 or you will face some issues with your DataLoader failing
    batch_size = 10
    n_epochs = 25

    # determine the loss and optimizer
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

    train_loader = DataLoader(transformed_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)
    # Start training
    print('\nStart training ...')
    train(n_epochs, train_loader)

    # save the model parameters
    model_dir = 'saved_models/'
    model_name = 'keypoints_model_AlexNet_50epochs.pth'
    torch.save(net.state_dict(), model_dir + model_name)
