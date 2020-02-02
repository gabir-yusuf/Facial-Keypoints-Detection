import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from models import AlexNet
from torch.autograd import Variable

import warnings

warnings.simplefilter("ignore")


def show_all_keypoints(img, key_pts):
    """
    Visualizing the image and the keypoints on it.
    """
    plt.imshow(img)
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=40, marker='.', c='g')
    plt.show()


def inference(model, img):
    """
    Forward propagate the input image through the model
    """
    # activate evaluation mode
    model.eval()

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img / 255.0

    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 0)

    img_torch = Variable(torch.from_numpy(img))
    img_torch = img_torch.type(torch.FloatTensor)
    return model(img_torch)


def add_sunglasses(glasses_img, key_pts, img_copy):
    # determine the top-left location for sunglasses which is the 17 point
    x = int(key_pts[17, 0])
    y = int(key_pts[17, 1])

    # setting the height and width of sunglasses
    # h = length of nose, w = left to right eyebrow edges
    h = int(abs(key_pts[27, 1] - key_pts[34, 1])) + 3
    w = int(abs(key_pts[17, 0] - key_pts[26, 0])) + 3

    # resize sunglasses
    new_sunglasses = cv2.resize(glasses_img, (w, h), interpolation=cv2.INTER_CUBIC)

    # get the region of interest on the face to change
    roi_color = img_copy[y:y + h, x + 5:x + w + 5]

    # find all non-transparent pts
    ind = np.argwhere(new_sunglasses[:, :, 3] > 0)

    # for each non-transparent point, replace the original image pixel with that of the new_sunglasses
    for i in range(3):
        roi_color[ind[:, 0], ind[:, 1],
                  i] = new_sunglasses[ind[:, 0], ind[:, 1], i]

    # set the area of the image to the changed region with sunglasses
    image_copy[y:y + h, x + 5:x + w + 5] = roi_color

    # display the result!
    plt.imshow(image_copy)
    plt.show()


def add_hat(hat_img, key_pts, img_copy):
    # determine the top-left location for hat
    x = int(key_pts[18, 0]) - 78
    y = int(key_pts[18, 1]) - 100

    # Determine the height and width of the hat
    h = 130
    w = 210

    # resize the hat
    hat_img = cv2.resize(hat_img, (w, h), interpolation=cv2.INTER_CUBIC)

    # get region of interest on the face to change
    roi_color = img_copy[y:y + h, x:x + w]

    # find all non-transparent pts
    ind = np.argwhere(hat_img[:, :, 3] > 0)

    for i in range(3):
        roi_color[ind[:, 0], ind[:, 1],
                  i] = hat_img[ind[:, 0], ind[:, 1], i]

    # set the area of the image to the changed region with hat
    img_copy[y:y + h, x:x + w] = roi_color

    # display the result!
    plt.imshow(img_copy)
    plt.show()


if __name__ == "__main__":
    net = AlexNet()
    image = cv2.imread('images/img.jpg')
    image = cv2.resize(image, (227, 227))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    net.load_state_dict(torch.load('./saved_models/keypoints_model.pth'))

    # inference
    keypoints = inference(net, image)

    # convert to numpy array, denormalize the keypoints, reshape to (2 x 68)
    keypoints = keypoints.data.numpy()
    keypoints = keypoints * 50.0 + 100
    keypoints = np.reshape(keypoints, (68, -1))

    image_copy = np.copy(image)
    show_all_keypoints(image_copy, keypoints)

    # Adding Sunglasses
    sunglasses_img = cv2.imread('images/sunglasses.png', cv2.IMREAD_UNCHANGED)
    add_sunglasses(sunglasses_img, keypoints, image_copy)

    # Adding Hat
    hat = cv2.imread('images/straw_hat.png', cv2.IMREAD_UNCHANGED)
    add_hat(hat, keypoints, image_copy)
