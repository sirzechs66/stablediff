import torch
from torch import conv2d, nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock,VAE_Residualblock

class VAE_Encoder(nn.Sequential):
    def ___it__(self):
        super().__init__(
            #(Batch_Size, Channel, Height, Width)->(Batch_Size, 128 , Height , Width)
            nn.Conv2d(3,128,kernel_size=3,padding=1),
            
            # (Batch_Size, 128, Height, width)-> (Batch_Size, 128, Height, width)
            VAE_Residualblock(128,128),
                  
            # (Batch_Size, 128, Height, width)-> (Batch_Size, 128, Height, width)
            VAE_Residualblock(128,128),
            
            # (Batch_size, 128, Height, width)->(Batch_size, 128, width /2, Height  /2)
            nn.Conv2d(128,128,kernel_size=3,stride=2,padding=0),
            
            # (Batch_Size, 128, Height, width)-> (Batch_Size, 256, Height, width)
            VAE_Residualblock(128,256),

            # (Batch_Size, 256, Height, width)-> (Batch_Size, 256, Height, width)
            VAE_Residualblock(256,256),

            #(Batch_Size,256, Channel, Height /2 , Width /2 )->(Batch_Size, 256 , Height /2, Width /2)  
            nn.Conv2d(256,256,kernel_size=3,stride=2,padding=0),
            
            # (Batch_Size, 256, Height /4, width /4)-> (Batch_Size, 512, Height /4, width /4)
            VAE_Residualblock(256,512),

            VAE_Residualblock(512,512),
            # (Batch_Size, 512, Height /4, width /4)-> (Batch_Size, 512, Height /4, width /4)

            # (Batch_size, 128, Height, width)->(Batch_size, 128, width /2, Height  /2)
            nn.Conv2d(512,512,kernel_size=3,stride=2,padding=0),
             

            VAE_Residualblock(512,512),

            VAE_Residualblock(512,512),
            
            # (Batch_Size, 512, Height /4, width /4)-> (Batch_Size, 512, Height /4, width /4)
            VAE_Residualblock(256,512),
            
            # (Batch_Size, 512, Height /4, width /4)-> (Batch_Size, 512, Height /4, width /4)
            VAE_AttentionBlock(512),
            
            # (Batch_Size, 512, Height /8, width /8)-> (Batch_Size, 512, Height /8, width /8)
            VAE_Residualblock(512,512),

            # (Batch_Size, 512, Height /8, width /8)-> (Batch_Size, 512, Height /8, width /8)
            nn.GroupNorm(32,512),

            # (Batch_Size, 512, Height /8, width /8)-> (Batch_Size, 512, Height /8, width /8)
            nn.SiLU(),

            # (Batch_Size, 512, Height /8, width /8)-> (Batch_Size, 8, Height /8, width /8)
            nn.Conv2d(512,8,kernel_size=3,padding=1),

            nn.Conv2d(512,8,kernel_size=3,padding=0)
        )
    def forward(self,x: torch.Tensor,noise:torch.Tensor)->torch.Tensor:
        # x:(batch_size, Channel, Height, Width) 
        # noise : (Batch_Size, Out_Channel, Height / 8,Width / 8) 
        for module in self:
            if getattr(module,'stride',None)==(2,2):
                # (Padding_Left, Padding_Left, Padding_Up, Padding_down)
                x = F.pad(x,(0,1,0,1)) 
            x = module(x)
         

         # (Batch_Size, 8, Height, Height / 8, Width / 8 )-> Two Tensors of Shape (Batch_size, 4 ,Height / 8, Width / 8)
        mean,log_variance = torch.chunk(x,2,dim=1)
        # (Batch_size, 4 ,Height / 8, Width / 8) -> (Batch_size, 4 ,Height / 8, Width / 8)
        log_variance = torch.clamp(log_variance,-30,20)

        #(Batch_size, 4 ,Height / 8, Width / 8) -> (Batch_size, 4 ,Height / 8, Width / 8)
        variance = log_variance.exp()

        #(Batch_size, 4 ,Height / 8, Width / 8) -> (Batch_size, 4 ,Height / 8, Width / 8)
        stdev = variance.sqrt()

        # Z=N(0,1) -> N(mean,Variance)=X?
        # Formula ++ > X = mean + stdev * Z(Noise)
        x=mean + stdev * noise 

        # Scale the Output by a constant
        x*= 0.18215
        
        return x

