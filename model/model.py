import torch
import torch.nn as nn


class PICNN(nn.Module):
    def __init__(self):
        super(PICNN, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=(16, 8), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size=(4, 4), stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512 * 14 * 7, 1024),
            nn.ReLU()
        )

        # Decoder for x-component
        self.decoder_x = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=(8, 8), stride=(2, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=(8, 4), stride=(2, 1), padding=(2, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, kernel_size=(4, 4), stride=2, padding=1)
        )

        # Decoder for y-component
        self.decoder_y = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=(8, 8), stride=(1, 2), padding=(0, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=(4, 8), stride=(1, 2), padding=(1, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, kernel_size=(4, 4), stride=2, padding=1)
        )

    def forward(self, x):
        # Encoder
        x = self.encoder(x)

        # Decoder
        x_component = self.decoder_x(x)
        y_component = self.decoder_y(x)

        return x_component, y_component
