import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class AttentionGated(nn.Module):
    def __init__(self,input_dim=1024,act='relu',bias=False,dropout=False, args=None):
        super(AttentionGated, self).__init__()
        self.L = args.input_dim
        self.D = 128 #128
        self.K = 1

        self.feature = [nn.Linear(args.input_dim, 512)]
        self.feature += [nn.ReLU()]
        self.feature += [nn.Dropout(0.25)]
        self.feature = nn.Sequential(*self.feature)

        self.classifier = nn.Sequential(  
            nn.Linear(512, args.n_classes),
        )

        self.attention_a = [
            nn.Linear(512, self.D,bias=bias),
        ]
        if act == 'gelu': 
            self.attention_a += [nn.GELU()]
        elif act == 'relu':
            self.attention_a += [nn.ReLU()]
        elif act == 'tanh':
            self.attention_a += [nn.Tanh()]

        self.attention_b = [nn.Linear(512, self.D,bias=bias),
                            nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(self.D, self.K,bias=bias)

        self.apply(initialize_weights)
    def forward(self, x):
        if x.dim() == 3:  # (batch_size, num_patches, feature_dim)
            batch_size, num_patches, feature_dim = x.shape
            batch_outputs = []
            for i in range(batch_size):
                x_i = x[i]  # [num_patches, feature_dim]
                x_i = self.feature(x_i)  # [num_patches, 512]
                
                a = self.attention_a(x_i)  # [num_patches, 128]
                b = self.attention_b(x_i)  # [num_patches, 128]
                A = a.mul(b)               # [num_patches, 128]
                A = self.attention_c(A)    # [num_patches, K]
                
                A = torch.transpose(A, -1, -2)  # [K, num_patches]
                A = F.softmax(A, dim=-1)        # [K, num_patches], softmax over num_patches
                
                x_i = x_i * A.reshape(-1, 1)    # [num_patches, 512]
                pred = self.classifier(x_i.sum(0).unsqueeze(0))  # [1, n_classes]
                batch_outputs.append(pred)
            
            pred = torch.cat(batch_outputs, dim=0)
        else:
            # 原始的单样本处理
            x = self.feature(x.squeeze(0))       # [num_patches, 512]
            x = x.squeeze(0)

            a = self.attention_a(x)              # [num_patches, 128]
            b = self.attention_b(x)              # [num_patches, 128]
            A = a.mul(b)                         # [num_patches, 128]
            A = self.attention_c(A)              # [num_patches, K]
            
            A = torch.transpose(A, -1, -2)       # [K, num_patches]
            A = F.softmax(A, dim=-1)             # [K, num_patches]
            
            x = x * A.reshape(-1, 1)             # [num_patches, 512]
            
            pred = self.classifier(x.sum(0).unsqueeze(0))  # [1, n_classes]
            
        Y_prob = F.softmax(pred, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]
        
        result_dict = {}
        
        return pred, Y_prob, Y_hat, None, result_dict

