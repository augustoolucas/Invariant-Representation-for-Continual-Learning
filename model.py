# Author: Ghada Sokar et al.
# This is the implementation for the Learning Invariant Representation for Continual Learning paper in AAAI workshop on Meta-Learning for Computer Vision

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def reparameterization(mu, logvar, latent_dim):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(
        Tensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))
    z = sampled_z * std + mu

    return z


class Encoder(nn.Module):

    def __init__(self, img_shape, n_hidden, latent_dim):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(n_hidden, latent_dim)
        self.logvar = nn.Linear(n_hidden, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar, self.latent_dim)

        return z, mu, logvar


class Decoder(nn.Module):

    def __init__(self,
                 img_shape,
                 n_hidden,
                 latent_dim,
                 n_classes,
                 use_label=True):
        super(Decoder, self).__init__()
        # conditional generation

        if use_label:
            input_dim = latent_dim + n_classes
        else:
            input_dim = latent_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, int(np.prod(img_shape))),
            nn.Sigmoid(),
        )
        self.img_shape = img_shape

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *self.img_shape)

        return img


class Specific(nn.Module):

    def __init__(self, img_shape, specific_size):
        super(Specific, self).__init__()

        # specific module
        self.specific = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), specific_size),
            #nn.ReLU(inplace=True),
        )

    def forward(self, imgs):
        x = self.specific(imgs.view(imgs.shape[0], -1))

        return x

class Classifier(nn.Module):

    def __init__(self,
                 latent_shape,
                 specific_shape,
                 n_classes,
                 classification_n_hidden=40):
        super(Classifier, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.classifier_layer = nn.Sequential(
            nn.Linear(specific_shape + latent_shape, classification_n_hidden),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Linear(classification_n_hidden, n_classes)

    def forward(self, discriminative, invariant):
        discriminative = self.relu(discriminative)
        x = self.classifier_layer(torch.cat([discriminative, invariant],
                                            dim=1))
        logits = self.output(x)

        return logits


class IRCLClassifier(nn.Module):

    def __init__(self, img_shape, invariant_n_hidden, specific_n_hidden,
                 classification_n_hidden, n_classes):
        super(IRCLClassifier, self).__init__()

        # specific module
        self.specific = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), specific_n_hidden),
            nn.ReLU(inplace=True))
        # classification module
        self.classifier_layer = nn.Sequential(
            nn.Linear(specific_n_hidden + invariant_n_hidden,
                      classification_n_hidden),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Linear(classification_n_hidden, n_classes)

    def forward(self, img, invariant):
        discriminative = self.specific(img)
        x = self.classifier_layer(torch.cat([discriminative, invariant],
                                            dim=1))
        logits = self.output(x)

        return logits
