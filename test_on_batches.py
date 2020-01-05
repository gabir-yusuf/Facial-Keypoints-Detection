import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import torch
from models import AlexNet
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data_load import FacialKeypointsDataset
from data_load import Rescale, RandomCrop, Normalize, ToTensor


import warnings
warnings.simplefilter("ignore")


net = AlexNet()
print(net)
# loading the best saved model parameters
net.load_state_dict(torch.load(
    './saved_models/keypoints_model.pth'))

# perepate the net for testing mode
net.eval()



batch_size = 10

data_transform = transforms.Compose([Rescale(250),
                                     RandomCrop(227),
                                     Normalize(),
                                     ToTensor()])

test_dataset = FacialKeypointsDataset(csv_file='data/test_frames_keypoints.csv',
                                      root_dir='data/test/',
                                      transform=data_transform)

test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=8)

# test the model on a batch of test images
def net_sample_output():
    # iterate through the batch dataset
    for i, sample in enumerate(test_loader):
        # get the sample data, images adn ground truth keypoints
        images, key_pts = sample['image'], sample['keypoints']
        n = 0
        for n in range(len(images)):
            img = images[n]
            img = np.transpose(img, (1, 2, 0))
            img = img.squeeze(2)
            print(img.shape)
            img = img * 255.0
            mpimg.imsave('./batch/' + str(n) + '.jpg', img,cmap='gray')


        # convert the images into float tensors
        images = images.type(torch.FloatTensor)

        # forward pass to get the net output
        output_pts = net(images)

        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)

        # break after the first batch
        if i == 0:
            return images, output_pts, key_pts


# visualize the predicted keypoints
def show_keypoints(image, predicted_key_pts, l):
    # Note >> image is grayscale
    plt.imshow(image, cmap='gray')

    # plot the predicted points
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=35, marker='.', c='m')

    # # plot the ground truth as green points
    # if gt_pts is not None:
    #     plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')
    plt.show()

# visualize the output
# by default this shows a batch of 10 images
def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):

    for i in range(batch_size):


        # un-transform the image data
        image = test_images[i].data   # get the image from it's wrapper
        image = image.numpy()   # convert to numpy array from a Tensor
        # transpose to go from torch to numpy image
        image = np.transpose(image, (1, 2, 0))

        # un-transform the predicted key_pts data
        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.numpy()
        # undo normalization of keypoints
        predicted_key_pts = predicted_key_pts*50.0 + 100

        # plot ground truth points for comparison, if they exist
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]
            ground_truth_pts = ground_truth_pts*50.0+100

        # call show_all_keypoints
        show_keypoints(np.squeeze(
            image), predicted_key_pts, l=i)







# ge a sample of the test data
test_images, test_output, gt_pts = net_sample_output()

# print out the shape of test batch
print(test_images.data.size())
print(test_output.data.size())
print(gt_pts.size())

# visualize the test reslt
visualize_output(test_images, test_output, gt_pts)
