import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, debug = True, input_shape = (90,90,3)):
        super().__init__(observation_space, features_dim=256)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.do_debug = debug
        self.debug = 0
        self.debug_iter = 0

        # CNN for image
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=6, stride=1, padding=0),  # (1, 114, 114) -> ...
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=6, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        # Compute CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_shape[2], *input_shape[:2])
            cnn_out_dim = self.cnn(dummy_input).view(-1).shape[0]

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim, 256),
            nn.ReLU()
        )

    def forward(self, observations):
        img = (observations['image'].float() / 255.0).to(self.device)

        if self.do_debug:
            self.debug_iter += 1
            self.debug += img.size()[0]
            if self.debug_iter % 50 == 0:
                print(f"\r{self.debug} images passed.", end= '')

        img_features = self.cnn(img)

        return self.fc(img_features)