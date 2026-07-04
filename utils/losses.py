from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
from utils.multi_similarity_loss_sij import MultiSimilarityLoss_Sij
import torch.nn.functional as F
import torch
import torch.nn as nn

def get_loss(loss_name):
    if loss_name == 'SupConLoss': return losses.SupConLoss(temperature=0.07)
    if loss_name == 'CircleLoss': return losses.CircleLoss(m=0.4, gamma=80) #these are params for image retrieval
    if loss_name == 'MultiSimilarityLossCM': return losses.MultiSimilarityLoss(alpha=2.0, beta=40, base=0.3, distance=DotProductSimilarity())
    if loss_name == 'MultiSimilarityLoss': return losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())
    if loss_name == 'MultiSimilarityLoss_Sij': return MultiSimilarityLoss_Sij(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())
    if loss_name == 'ContrastiveLoss': return losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    if loss_name == 'Lifted': return losses.GeneralizedLiftedStructureLoss(neg_margin=0, pos_margin=1, distance=DotProductSimilarity())
    if loss_name == 'FastAPLoss': return losses.FastAPLoss(num_bins=30)
    if loss_name == 'NTXentLoss': return losses.NTXentLoss(temperature=0.07) #The MoCo paper uses 0.07, while SimCLR uses 0.5.
    if loss_name == 'TripletMarginLoss': return losses.TripletMarginLoss(margin=0.1, swap=False, smooth_loss=False, triplets_per_anchor='all') #or an int, for example 100
    if loss_name == 'CentroidTripletLoss': return losses.CentroidTripletLoss(margin=0.05,
                                                                            swap=False,
                                                                            smooth_loss=False,
                                                                            triplets_per_anchor="all",)
    raise NotImplementedError(f'Sorry, <{loss_name}> loss function is not implemented!')

def get_miner(miner_name, margin=0.1):
    if miner_name == 'TripletMarginMiner' : return miners.TripletMarginMiner(margin=margin, type_of_triplets="semihard") # all, hard, semihard, easy
    if miner_name == 'MultiSimilarityMiner' : return miners.MultiSimilarityMiner(epsilon=margin, distance=CosineSimilarity())
    if miner_name == 'PairMarginMiner' : return miners.PairMarginMiner(pos_margin=0.7, neg_margin=0.3, distance=DotProductSimilarity())
    return None


def contrastive_loss_cross_modal(image_features, text_features, temp):
    """
    Computes symmetric contrastive loss for image-text pairs.
    - image_features: [batch_size, embedding_dim]
    - text_features: [batch_size, embedding_dim]
    - logit_scale: learnable scalar (e.g., e^4.6052)
    """
    with torch.no_grad():
        temp.clamp_(0.01,0.5)
    # Compute similarity matrix (logits)
    logits_per_image = (image_features @ text_features.T) / temp
    logits_per_text = logits_per_image.T

    # Create ground truth labels (diagonal elements)
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)

    # Calculate symmetric cross-entropy loss
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss = (loss_i + loss_t) / 2
    #loss = loss_t
    
    return loss

class MultiSimilarityLossForPairwiseScores(nn.Module):
    def __init__(self, alpha=2.0, beta=50.0, lambda_val=0.5):
        super().__init__()
        self.alpha = alpha     # Gradient scale for weak positives
        self.beta = beta       # Gradient scale for hard negatives (focuses heavily on tough VPR failures)
        self.lambda_val = lambda_val

    def forward(self, score_matrix, labels):
        """
        score_matrix: [B, B] tensor containing the raw scalar matching logits from your MLP.
                      score_matrix[i, j] = Score(Text_i, Image_j)
        labels:       [B] tensor containing structural location class tags (e.g., [0,0,0,0, 1,1,1,1, ...])
        """
        batch_size = score_matrix.size(0)
        loss = 0.0
        
        # 1. Generate the absolute alignment mask grid [B, B]
        # entry [i, j] is True if index i and index j belong to the same physical place
        adjacency_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1))
        
        # Disregard the diagonal identity (self-matching elements) from pair optimization
        identity_mask = torch.eye(batch_size, dtype=torch.bool, device=score_matrix.device)
        
        # Isolate true companion positives and explicit out-of-class negatives
        pos_mask = adjacency_matrix & ~identity_mask
        neg_mask = ~adjacency_matrix

        # 2. Compute soft-mining exponential margins across rows
        for i in range(batch_size):
            row_scores = score_matrix[i]
            
            # Extract the 3 structural positive matching scores and all out-of-class negative blocks
            pos_scores = row_scores[pos_mask[i]]
            neg_scores = row_scores[neg_mask[i]]
            
            # Guard rail for edge configurations
            if len(pos_scores) == 0 or len(neg_scores) == 0:
                continue
            
            # Natural soft mining via Multi-Similarity log-exponential pooling:
            # Positives close to 0 receive high gradients; high negative scores explode exponentially
            pos_term = (1.0 / self.alpha) * torch.log(1.0 + torch.sum(torch.exp(-self.alpha * (pos_scores - self.lambda_val))))
            neg_term = (1.0 / self.beta) * torch.log(1.0 + torch.sum(torch.exp(self.beta * (neg_scores - self.lambda_val))))
            
            loss += pos_term + neg_term
            
        return loss / batch_size