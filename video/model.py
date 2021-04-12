import torch
import torch.nn as nn


class FaceRecognizer(nn.Module):

    def __init__(self, feature, classnum):
        super(FaceRecognizer, self).__init__()
        self.feature = feature
        self.classifier = nn.Sequential(*[
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, classnum)
        ])

    def load_feature(self, file_path):
        self.feature.load_state_dict(torch.load(file_path)['net_state_dict'])

    def train(self, mode=True):
        self.feature.eval()
        self.training = mode
        for module in self.classifier.children():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x):
        out = self.feature(x)
        logit = self.classifier(out)
        return logit
