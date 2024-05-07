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

    def physics_loss(self, u_pred, v_pred, u_gt, v_gt, rho=1.0, mu=1.0):
        # Compute data loss
        data_loss = nn.MSELoss()(u_pred, u_gt) + nn.MSELoss()(v_pred, v_gt)

        # Compute physics residuals
        u_x = torch.gradient(u_pred, dim=(2, 3))[0]
        v_y = torch.gradient(v_pred, dim=(2, 3))[1]
        R_c = u_x + v_y

        u_t = torch.zeros_like(u_pred)  # Assuming steady-state
        u_xx = torch.gradient(torch.gradient(u_pred, dim=3)[0], dim=3)[0]
        u_yy = torch.gradient(torch.gradient(u_pred, dim=2)[0], dim=2)[0]
        v_xx = torch.gradient(torch.gradient(v_pred, dim=3)[0], dim=3)[0]
        v_yy = torch.gradient(torch.gradient(v_pred, dim=2)[0], dim=2)[0]
        R_u = rho * (u_t + u_pred * u_x + v_pred * u_y) + torch.ones_like(u_pred) - mu * (u_xx + u_yy)
        R_v = rho * (v_t + u_pred * v_x + v_pred * v_y) + torch.ones_like(v_pred) - mu * (v_xx + v_yy)

        physics_residual = torch.mean(R_c ** 2 + R_u ** 2 + R_v ** 2)

        alpha = 0.7 # Data loss weight
        beta = 0.3 # Physics residual loss weight
        loss = alpha * data_loss + beta * physics_residual

        return loss
