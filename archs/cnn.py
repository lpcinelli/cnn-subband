import torch.nn as nn
import torch.nn.functional as F
from utils import construct_wt_filters, wt


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @classmethod
    def from_name(cls, num_classes, **kwargs):
        # cls._check_model_name_is_valid(model_name)
        # blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(nb_classes)

# Original:
# init biases to 1
# init weights from N(0, 0.01^2)

# Here (default):
# init bias as Lecunn: U(-k,k) w/k = sqrt(1/fan_in)
# init weights as kaiming_uniform: U(-k,k) w/k = sqrt(6/fan_in)
