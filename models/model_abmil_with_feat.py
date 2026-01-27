import torch
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_abmil import AttentionGated

class AttentionGatedWithFeat(AttentionGated):
    def forward(self, x):
        if x.dim() == 3:  # (batch_size, num_patches, feature_dim)
            batch_size, num_patches, feature_dim = x.shape
            batch_outputs = []
            batch_features = []
            for i in range(batch_size):
                x_i = x[i]  # (num_patches, feature_dim)
                x_i = self.feature(x_i)  # (num_patches, 512)
                a = self.attention_a(x_i)
                b = self.attention_b(x_i)
                A = a.mul(b)
                A = self.attention_c(A)
                A = torch.transpose(A, -1, -2)  # KxN
                A = F.softmax(A, dim=-1)  # softmax over N
                x_weighted = x_i * A.reshape(-1, 1)
                wsi_feature = x_weighted.sum(0)  # (512,)
                pred = self.classifier(wsi_feature.unsqueeze(0))
                batch_outputs.append(pred)
                batch_features.append(wsi_feature.unsqueeze(0))
            pred = torch.cat(batch_outputs, dim=0)
            wsi_feature = torch.cat(batch_features, dim=0)  # (batch_size, 512)
        else:
            x = self.feature(x.squeeze(0))
            x = x.squeeze(0)
            a = self.attention_a(x)
            b = self.attention_b(x)
            A = a.mul(b)
            A = self.attention_c(A)
            A = torch.transpose(A, -1, -2)  # KxN
            A = F.softmax(A, dim=-1)  # softmax over N        
            x_weighted = x * A.reshape(-1, 1)
            wsi_feature = x_weighted.sum(0)  # (512,)
            pred = self.classifier(wsi_feature.unsqueeze(0))
        Y_prob = F.softmax(pred, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]
        result_dict = {}
        return pred, Y_prob, Y_hat, wsi_feature, result_dict 