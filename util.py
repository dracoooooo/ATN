import torch
import torch.nn as nn


# cnn for mnist
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=0),  # [64, 24, 24]
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 12, 12]

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0),  # [128, 8, 8]
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 512),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


def r(y, t, a):
    m = y.max(dim=1)[0]
    res = torch.clone(y)
    res[:, t] = a * m
    return res


def l(x, generate, pred_raw, pred_generate, b, t, r):
    loss = nn.MSELoss()
    Lx = b * loss(generate, x)
    Ly = loss(pred_generate, r(pred_raw, t, 2))
    return Lx + Ly
