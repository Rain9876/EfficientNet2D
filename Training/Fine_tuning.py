from collections import OrderedDict
# from EfficientNet_Model.Model import EfficientNetB0
# import os
import sys
# sys.path.insert(0,"/Users/yurunsong/EfficientNet2D/")
# print(sys.path、

# from ray import tune
# from ray.tune.examples.mnist_pytorch import get_data_loaders, ConvNet, train, test

from Pre_trained_Model.model_ import EfficientNet
from Pre_trained_Model.utils_ import get_model_params,load_pretrained_weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Training.Model_Training import training, testing, ImageProcessing
import torch
import torch.nn
import matplotlib.pyplot as plt
from PIL import Image
import json

torch.manual_seed(0)
use_GPU = torch.cuda.is_available()
device = torch.device("cuda" if use_GPU else "cpu")
if use_GPU: torch.cuda.manual_seed(0)
print("Using GPU: {}".format(use_GPU))

img_stats = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
resolution = 240


def pre_trained_model():
    # model = EfficientNetB0(1000)

    # block_args, global_args = efficientnet(width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2,
    #                  drop_connect_rate=0.2, image_size=resolution, num_classes=1000)

    block_args, global_args = get_model_params("efficientnet-b0", None)

    model = EfficientNet(block_args, global_args)

    instance = {
        "b0": "efficientnet-b0-08094119.pth",
        "b1": "efficientnet-b1-dbc7070a.pth",
        "b2": "efficientnet-b2-27687264.pth",
        "b3": "efficientnet-b3-c8376fa2.pth",
        "b4": "efficientnet-b4-e116e8b3.pth",
        "b5": "efficientnet-b5-586e6cc6.pth",
        "b6": "efficientnet-b6-c76e70fd.pth",
        "b7": "efficientnet-b7-dcc49843.pth",
    }

    model_state = torch.load(sys.path[0]+"/data/Pre_trained_states/" + instance["b0"])
    model.load_state_dict(model_state, strict=False)

    #
    # # A basic remapping is required
    # mapping = {
    #     k: v for k, v in zip(model_state.keys(), model.state_dict().keys())
    # }
    # mapped_model_state = OrderedDict([
    #     (mapping[k], v) for k, v in model_state.items()
    # ])
    #


    return model


def imageShow(path):
    img = Image.open(path)
    # Preprocess image
    tfms = transforms.Compose([transforms.Resize([resolution] * 2),
                               transforms.ToTensor(),
                               transforms.Normalize(*img_stats), ])
    x = tfms(img).unsqueeze(0)

    print(x.shape)
    plt.imshow(img)
    plt.show()
    return x


def subTestForPreTraining(model, image):
    with open("../Pre_trained_Model/labels_map.txt", "r") as h:
        labels = json.load(h)

    # Classify
    model.eval()
    with torch.no_grad():
        y_pred = model(image)

    # Print predictions
    print('-----')
    for idx in torch.topk(y_pred, k=5)[1].squeeze(0).tolist():
        prob = torch.softmax(y_pred, dim=1)[0, idx].item()
        print('{label:<75} ({p:.2f}%)'.format(label=labels[str(idx)], p=prob * 100))


#### Fine-tuning

def ImageProcessingForFolder(path, batch_size, img_size, img_stats):
    transform = transforms.Compose([transforms.Resize(img_size), transforms.CenterCrop(img_size),
                                    transforms.ToTensor(), transforms.Normalize(img_stats)])

    train_dat = datasets.ImageFolder(root=path + "/train", transform=transform)

    train_loader = DataLoader(train_dat, batch_size=batch_size, shuffle=True, num_workers=2)

    test_dat = datasets.ImageFolder(root=path + './test', transform=transform)

    test_loader = torch.utils.data.DataLoader(test_dat, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


classes = {0: 'No DR', 1: 'Mild DR', 2: 'Moderate DR', 3: 'Severe DR', 4: 'Poliferative DR'}
classes_binary = {0: 'No DR', 1: 'DR'}


# No longer adjust the pretained parameters if True
def set_parameter_requires_grad(model, feature_extract=None):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False


def updateParameters(model, feature_extract=False):
    params_to_update = model.parameters()

    # print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
    #             print("\t", name)
    # else:
    #     for name, param in model.named_parameters():
    #         if param.requires_grad == True:
    #             print("\t", name)

    return params_to_update


def fine_tuning(model, num_classes):

    epochs = 10

    set_parameter_requires_grad(model)

    input_features = model._fc.in_features

    output_features = num_classes

    model._fc = torch.nn.Linear(input_features, output_features, bias=False)

    model.to(device)

    params_to_update = updateParameters(model)

    # optimizer = torch.optim.RMSprop(params_to_update, lr = 0.0001, weight_decay =1e-5)
    optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.99, weight_decay=1e-5)
    # optimizer = torch.optim.SGD(params_to_update, lr=config["lr"], momentum=config["momentum"])

    # train_loader, test_loader = ImageProcessingForFolder("****************", 32, resolution, img_stats)

    train_loader, test_loader = ImageProcessing()

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):

        training(model, train_loader, optimizer, criterion, epoch)

        acc = testing(model, test_loader, criterion)

    return model








#
# import ray
# from ray import tune
# from ray.tune import track
# from ray.tune.schedulers import AsyncHyperBandScheduler
# import numpy as np
#
# def gridSearch(config):
#
#     md = pre_trained_model()
#
#     model = fine_tuning(md, 10)
#
#     params_to_update = updateParameters(model)
#
#     # optimizer = torch.optim.RMSprop(params_to_update, lr = 0.0001, weight_decay =1e-5)
#     # optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.99, weight_decay=1e-5)
#     optimizer = torch.optim.SGD(params_to_update, lr=config["lr"], momentum=config["momentum"])
#
#     # train_loader, test_loader = ImageProcessingForFolder("****************", 32, resolution, img_stats)
#
#     train_loader, test_loader = ImageProcessing()
#
#     criterion = torch.nn.CrossEntropyLoss()
#
#     # for epoch in range(epochs):
#     i = 0
#     while True:
#
#         training(model, train_loader, optimizer, criterion, i)
#
#         acc = testing(model, test_loader, criterion)
#
#         i+=1
#
#         track.log(mean_accuracy=acc)
#
#
#
# sched = AsyncHyperBandScheduler(time_attr="training_iteration", metric="mean_accuracy")
# analysis = tune.run(
#         gridSearch,
#         name="exp",
#         scheduler=sched,
#         stop={
#             "mean_accuracy": 0.97,
#             "training_iteration": 15
#         },
#         config={
#             "lr": tune.sample_from(lambda spec: 10**(-10 * np.random.rand())),
#             "momentum": tune.uniform(0.1, 0.9),
#         })
#
# print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))
#


# md = pre_trained_model()
# img = imageShow("../data/img.jpg")
# subTestForPreTraining(md, img)
# fine_tuning(md, 10)