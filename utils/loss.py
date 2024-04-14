import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class ContrastiveLoss(nn.Module):

    def __init__(self, margin, similaritylayer):
        super().__init__()

        self.similaritylayer = similaritylayer
        self.margin = margin

        
    def forward(self, d_tot, labels):

        label_similarity = torch.sum(labels[:len(labels)//2] * labels[len(labels)//2:], dim=1)
        loss = torch.mean((1 - label_similarity) * torch.pow(d_tot, 2) + label_similarity * torch.pow(torch.clamp(self.margin - d_tot, min=0.0), 2))

        return loss

class WeightedLoss(nn.Module):
    def __init__(self, config, bit):
        super(WeightedLoss, self).__init__()
        self.scale = 1

    def forward(self, u1, u2, y, config):

        sigmoid_alpha = config["alpha"]
        u1 = torch.tanh(self.scale * u1)
        u2 = torch.tanh(self.scale * u2)

        y1 = y[:len(y)//2]
        y2 = y[len(y)//2:]

        y_S = (y1 @ y2.t() > 0).float()       
        
        dot_product = sigmoid_alpha * u1 @ u2.t()

        mask_positive = y_S > 0
        mask_negative = (1 - y_S).bool()
        neg_log_probe = dot_product + torch.log(1 + torch.exp(-dot_product)) -  y_S * dot_product
        S1 = torch.sum(mask_positive.float())
        S0 = torch.sum(mask_negative.float())
        S = S0 + S1
        
        neg_log_probe[mask_positive] = neg_log_probe[mask_positive] * S / S1
        neg_log_probe[mask_negative] = neg_log_probe[mask_negative] * S / S0
        loss = torch.sum(neg_log_probe) / S

        return loss

class CosineSimilarity(nn.Module):
    """Computes the cosine similarity between embeddings
    """
    def __init__(self):
        super().__init__()

    def forward(self, emb_i, emb_j):

        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        Z = torch.cat([z_i, z_j], dim=0)

        return torch.matmul(Z, torch.t(Z))
        
