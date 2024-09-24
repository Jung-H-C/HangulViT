import torch.nn as nn
import torch


def flatten_features(x):
    batch_size, channels, height, width = x.size()
    feature_map = x.view(batch_size, channels, height * width)
    feature_map = feature_map.permute(0, 2, 1)

    return feature_map


class CNNEncoder(nn.Module):
    def __init__(self, embed_dim = 512, num_patches = 121):
        super(CNNEncoder, self).__init__()
        # input_size: [batch_size, 1, 360, 360]
        # desired_output_size: [batch_size, 121, 512]

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # [batch_size, 64, 180, 180]
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # [batch_size, 128, 90, 90]
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # [batch_size, 256, 45, 45]
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4) # [batch_size, 512, 11, 11]
        )

        self.pos_embedding = nn.Parameter(torch.rand(1, num_patches, embed_dim))

    def forward(self, image):
        out = self.conv1(image)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        features = flatten_features(out)
        features += self.pos_embedding[:, :features.size(1), :]

        return features # [batch_size, 121, 512]



# random_image = torch.randn(32, 1, 360, 360)
#
# print(CNNEncoder().forward(random_image))
# print((CNNEncoder().forward(random_image)).size())