import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


def is_cancer(file_path):

    transform = transforms.Compose([transforms.Resize(225), transforms.CenterCrop(224),
                                    transforms.ToTensor(), transforms.Normalize((.5, .5, .5), (.5, .5, .5))])

    classes = ['benign', 'malignant']

    model = torch.load('cancer_model.pth', map_location=torch.device('cpu'))

    img = Image.open(file_path)
    img = transform(img).unsqueeze(0)
    out = model(img)
    top_k, top_class = out.topk(1, dim=1)
    with torch.no_grad():
        prob = torch.exp(F.log_softmax(out, dim=1))
    # print(prob)
    d = dict()
    for i in range(len(classes)):
        d[classes[i]] = prob[0][i].item()
    return classes[top_class], d


if __name__ == "__main__":
    f_name = input("Enter the file name : ")
    res, prob = is_cancer((f_name))
    print(res)
    print(prob)
