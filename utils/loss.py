import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class ContrastiveLoss(nn.Module):

    def __init__(self, batchsize, margin, similaritylayer, num_worker=4):
        super().__init__()

        self.batchsize = batchsize
        self.similaritylayer = similaritylayer
        self.margin = margin
        self.num_worker = num_worker
        
    def forward(self, emb_i, emb_j):

        similarities1 = self.similaritylayer(emb_i, emb_j)
        similarities = torch.exp(similarities1/self.margin)
        
        # get similarities between anchor-positive pairs
        pos_zij = torch.diag(similarities, self.batchsize*self.num_worker)
        pos_zji = torch.diag(similarities, -self.batchsize*self.num_worker)
        numerator = torch.cat([pos_zij, pos_zji],dim=0)

        # get similarities between anchor-negative pairs
        mask = ~torch.diag(torch.ones(2*self.batchsize*self.num_worker)).bool()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask = mask.to(device)

        # print(self.num_worker, similarities.shape, 2*self.batchsize*self.num_worker)
        denominator =  torch.sum(torch.masked_select(similarities, mask).view(2*self.batchsize*self.num_worker, 2*self.batchsize*self.num_worker-1), dim=1)
        loss = torch.mean(-torch.log(numerator/denominator))

        return loss, similarities1

class WeightedLoss(nn.Module):
    def __init__(self, config, bit):
        super(WeightedLoss, self).__init__()
        self.scale = 1

    def forward(self, u, y, ind, config):
        u = torch.tanh(self.scale * u)
        S = (y @ y.t() > 0).float()       
        sigmoid_alpha = config["alpha"]
        dot_product = sigmoid_alpha * u @ u.t()   
        mask_positive = S > 0
        mask_negative = (1 - S).bool()    
        neg_log_probe = dot_product + torch.log(1 + torch.exp(-dot_product)) -  S * dot_product
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
        
