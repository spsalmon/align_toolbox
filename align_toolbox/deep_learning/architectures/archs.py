import torch
from torch import nn


# 1D UNet implementation
class VGGBlock1D(nn.Module):
    """
    One-dimensional two-layer convolutional block with BatchNorm and ReLU.

    The 1D analogue of :class:`VGGBlock`, using Conv1d. Used as the basic
    building block in :class:`Unet1D`, :class:`AttentionUnet1D`, and
    :class:`UnetPlusPlus1D`.

    Parameters:
        in_channels (int): Number of input channels.
        middle_channels (int): Number of channels after the first convolution.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(middle_channels)
        self.conv2 = nn.Conv1d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class DilatedBottleneck1D(nn.Module):
    """Expand receptive field at the bottleneck without changing resolution."""

    def __init__(self, channels, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, padding=d, dilation=d),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
            )
            for d in dilations
        )

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)  # residual, so it is a safe add-on to a trained-ish net
        return x


class Unet1D(nn.Module):
    """
    1D U-Net for sequence keypoint detection with a per-channel presence head.

    Returns a tuple ``(heatmap_logits, presence_logits)``:
      - heatmap_logits: [B, num_classes, T]  -- localization ("where")
      - presence_logits: [B, num_classes]    -- existence ("whether"), raw logits

    Filter counts follow [32, 64, 128, 256, 512].

    Parameters:
        num_classes (int): Number of keypoint channels.
        input_channels (int, optional): Number of input sequence channels. (default: 1)
        **kwargs: Ignored; accepted for API compatibility.
    """

    def __init__(self, num_classes, input_channels=1, **kwargs):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool1d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)

        self.bottleneck = DilatedBottleneck1D(nb_filter[4])

        self.conv0_0 = VGGBlock1D(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock1D(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock1D(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock1D(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock1D(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock1D(
            nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3]
        )
        self.conv2_2 = VGGBlock1D(
            nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2]
        )
        self.conv1_3 = VGGBlock1D(
            nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1]
        )
        self.conv0_4 = VGGBlock1D(
            nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0]
        )

        self.final = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1)

        # presence head: per-channel "does this molt exist" logit.
        # fed by a masked mean over valid frames of the final decoder features.
        self.presence_head = nn.Sequential(
            nn.Linear(nb_filter[0], nb_filter[0]),
            nn.ReLU(inplace=True),
            nn.Linear(nb_filter[0], num_classes),
        )

    def forward(self, input, mask=None):
        """
        input : [B, input_channels, T]
        mask  : [B, T] or None. 1 = valid frame, 0 = padding. Used only to keep
                padded frames out of the presence pooling. If None, a plain mean
                over all frames is used (assumes no padding).
        """
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.bottleneck(self.conv4_0(self.pool(x3_0)))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))  # [B, nb_filter[0], T]

        heatmap_logits = self.final(x0_4)  # [B, num_classes, T]

        # masked mean over time -> [B, nb_filter[0]]
        if mask is not None:
            m = mask.to(x0_4.dtype).unsqueeze(1)  # [B, 1, T]
            pooled = (x0_4 * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1.0)
        else:
            pooled = x0_4.mean(dim=-1)
        presence_logits = self.presence_head(pooled)  # [B, num_classes]

        return heatmap_logits, presence_logits
