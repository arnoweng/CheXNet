import numpy as np
import torch
from torch.autograd import Variable
import cv2
from imagenet_index import index2class
import torch.nn.functional as F

def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def normalize(Ac):
    Ac_shape = Ac.shape
    AA = Ac.view(Ac.size(0), -1)
    AA -= AA.min(1, keepdim=True)[0]
    AA /= AA.max(1, keepdim=True)[0]
    scaled_ac = AA.view(Ac_shape)
    return scaled_ac

def tensor2image(x, i =0):
    x = normalize(x)
    x = x[i].detach().cpu().numpy()
    x = cv2.resize(np.transpose(x, (1, 2, 0)), (224, 224))
    return x

def threshold(x):
    mean_ = x.mean()
    std_ = x.std()
    thresh = mean_ +std_
    x = (x>thresh)
    return x


class ScoreCAM(object):

    """
        ScoreCAM, inherit from BaseCAM

    """

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = dict()
        self.activations = dict()

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        def backward_hook(module,input,output):
            self.gradients['value'] = output[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        score_weight = []
        # predication on raw input
        logit = self.model(input).cuda()

        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()

        logit = F.softmax(logit)

        if torch.cuda.is_available():
          predicted_class= predicted_class.cuda()
          score = score.cuda()
          logit = logit.cuda()

        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        activations = self.activations['value']
        b, k, u, v = activations.size()

        score_saliency_map = torch.zeros((1, 1, h, w))

        if torch.cuda.is_available():
          activations = activations.cuda()
          score_saliency_map = score_saliency_map.cuda()

        with torch.no_grad():
          for i in range(k):

              # upsampling
              saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
              saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

              if saliency_map.max() == saliency_map.min():
                  score_weight.append(0)
                  continue

              # normalize to 0-1
              norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

              # how much increase if keeping the highlighted region
              # predication on masked input
              output = self.model(input * norm_saliency_map)
              output = F.softmax(output)
              score = output[0][predicted_class]

              score_saliency_map += score * saliency_map
              score_weight.append(score.detach().cpu().numpy())

        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data

        return score_saliency_map, score_weight

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)




