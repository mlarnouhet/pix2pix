import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_channels, ngf, out_channels):
        super(Generator, self).__init__()
        #Encoder layers
        self.e1 = nn.Sequential(
            nn.Conv2d(in_channels, ngf, 4, 2, 1),
            nn.LeakyReLU(0.2))
        self.e2 = self.conv_block(ngf, ngf*2, 4, 2, 1)
        self.e3 = self.conv_block(ngf*2, ngf*4, 4, 2, 1)
        self.e4 = self.conv_block(ngf*4, ngf*8, 4, 2, 1)
        self.e5 = self.conv_block(ngf*8, ngf*8, 4, 2, 1)
        self.e6 = self.conv_block(ngf*8, ngf*8, 4, 2, 1)
        self.e7 = self.conv_block(ngf*8, ngf*8, 4, 2, 1)
        self.e8 = nn.Sequential(
            nn.Conv2d(
                ngf*8,
                ngf*8,
                4,
                2,
                1,
                bias=False,
            ),
            nn.ReLU(),
        )

        #Decoder layers
        self.d1 = self.deconv_block(ngf*8, ngf*8, 4, 2, 1)
        self.dp1 = nn.Dropout(0.5)
        self.d2 = self.deconv_block(ngf*8*2, ngf*8, 4, 2, 1)
        self.dp2 = nn.Dropout(0.5)
        self.d3 = self.deconv_block(ngf*8*2, ngf * 8, 4, 2, 1)
        self.dp3 = nn.Dropout(0.5)
        self.d4 = self.deconv_block(ngf*8*2, ngf*8, 4, 2, 1)
        self.d5 = self.deconv_block(ngf*8*2, ngf*4, 4, 2, 1)
        self.d6 = self.deconv_block(ngf*4*2, ngf*2, 4, 2, 1)
        self.d7 = self.deconv_block(ngf*2*2 , ngf, 4, 2, 1)
        self.d8 = nn.Sequential(
            nn.ConvTranspose2d(
                ngf*2,
                out_channels,
                4,
                2,
                1,
                bias=False,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        x5 = self.e5(x4)
        x6 = self.e6(x5)
        x7 = self.e7(x6)
        x8 = self.e8(x7)

        y1 = self.dp1(self.d1(x8))
        y1 = torch.concatenate((y1,x7), 1)
        y2 = self.dp2(self.d2(y1))
        y2 = torch.concatenate((y2, x6), 1)
        y3 = self.dp3(self.d3(y2))
        y3 = torch.concatenate((y3, x5), 1)

        y4 = self.d4(y3)
        y4 = torch.concatenate((y4,x4), 1)
        y5 = self.d5(y4)
        y5 = torch.concatenate((y5, x3), 1)
        y6 = self.d6(y5)
        y6 = torch.concatenate((y6, x2), 1)
        y7 = self.d7(y6)
        y7 = torch.concatenate((y7,x1), 1)
        y8 = self.d8(y7)
        #y8 = torch.tanh(y8)

        return y8

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
        )

    def deconv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )



class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, ndf):
        super(Discriminator, self).__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(
                in_channels+out_channels,
                ndf,
                4,
                2,
                1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, True),
        )

        self.c2 = self.conv_block(ndf, ndf*2, 4, 2, 1)
        self.c3 = self.conv_block(ndf*2, ndf*4, 4, 2, 1)
        self.c4 = self.conv_block(ndf*4, ndf*8, 4, 1, 1)
        self.c5 = nn.Conv2d(ndf*8, 1, 4, 1, 1)
        #self.s = nn.Sigmoid()

    def forward(self, x, y):
        z = self.c1(torch.concatenate((x,y), 1))
        z = self.c2(z)
        z = self.c3(z)
        z = self.c4(z)
        z = self.c5(z)
        #z = self.s(z)

        return z


    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
        )


def test():
    batch_size = 10
    gen = Generator(3,64,3)
    disc = Discriminator(3, 3, 64)
    input = torch.randn(batch_size,3,256,256)
    output = gen(input)
    score = disc(torch.concatenate((output, output), 1))

if __name__ == "__main__":
    test()