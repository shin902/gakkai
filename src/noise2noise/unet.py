import torch
nn =  torch.nn

class Unet(nn.Module):
    def __init__(self,cn=3):
        super(Unet,self).__init__()

        # copu = convolution + pooling
        self.copu1 = nn.Sequential(
            nn.Conv2d(cn,48,3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48,48,3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        for i in range(2,6): #copu2, copu3, copu4, copu5
            self.add_module('copu%d'%i,
                nn.Sequential(
                    nn.Conv2d(48,48,3,stride=1,padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2)
                )
            )

        # coasa = convolution + upsample
        self.coasa1 = nn.Sequential(
            nn.Conv2d(48,48,3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48,48,3,stride=2,padding=1,output_padding=1)
        )

        self.coasa2 = nn.Sequential(
            nn.Conv2d(96,96,3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96,96,3,stride=2,padding=1,output_padding=1)
        )

        for i in range(3,6): #coasa3, coasa4, coasa5
            self.add_module('coasa%d'%i,
                nn.Sequential(
                    nn.Conv2d(144,96,3,stride=1,padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(96,96,3,stride=1,padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(96,96,3,stride=2,padding=1,output_padding=1)
                )
            )

        # coli = convolution + leakyrelu
        self.coli = nn.Sequential(
            nn.Conv2d(96+cn,64,3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,32,3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,cn,3,stride=1,padding=1),
            nn.LeakyReLU(0.1)
        )

        # 重みの初期値
        for l in self.modules():
            if(type(l) in (nn.ConvTranspose2d,nn.Conv2d)):
                nn.init.kaiming_normal_(l.weight.data)
                l.bias.data.zero_()

    def forward(self,x):
        x1 = self.copu1(x)
        x2 = self.copu2(x1)
        x3 = self.copu3(x2)
        x4 = self.copu4(x3)
        x5 = self.copu5(x4)

        z = self.coasa1(x5)
        z = self.coasa2(torch.cat((z,x4),1))
        z = self.coasa3(torch.cat((z,x3),1))
        z = self.coasa4(torch.cat((z,x2),1))
        z = self.coasa5(torch.cat((z,x1),1))

        return self.coli(torch.cat((z,x),1))

