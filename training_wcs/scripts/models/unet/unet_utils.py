import torch
import torch.nn as nn
import torch.nn.functional as F


class UnetConv2(nn.Module):

    def __init__(self, in_channels, out_channels, is_batchnorm):
        super(UnetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                # this amount of padding/stride/kernel_size preserves width/height
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetUp(nn.Module):

    def __init__(self, in_channels, out_channels, is_deconv):
        """

        is_deconv:  use transposed conv layer to upsample - parameters are learnt; otherwise use
                    bilinear interpolation to upsample.
        """
        super(UnetUp, self).__init__()

        self.conv = UnetConv2(in_channels, out_channels, False)

        self.is_deconv = is_deconv
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # nn.UpsamplingBilinear2d is deprecated in favor of F.interpolate()
        # else:
        #     self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        """
        inputs1 is from the downward path, of higher resolution
        inputs2 is from the 'lower' layer. It gets upsampled (spatial size increases) and its depth (channels) halves
        to match the depth of inputs1, before being concatenated in the depth dimension.
        """
        if self.is_deconv:
            outputs2 = self.up(inputs2)
        else:
            # scale_factor is the multiplier for spatial size
            outputs2 = F.interpolate(inputs2, scale_factor=2, mode='bilinear', align_corners=True)

        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)

        return self.conv(torch.cat([outputs1, outputs2], dim=1))
