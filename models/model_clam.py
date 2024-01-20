import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.nn import Parameter 
from torch.autograd import Variable 
from utils.utils import initialize_weights
import numpy as np


# Define the CenterLoss criterion
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, centers, alpha=0.5):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha
        self.centers = centers # Size [num_classes x feat_dim]

    def forward(self, h, labels):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        labels = torch.tensor([labels] * h.size(0))  # Repeat the scalar label for each sample in the batch
        labels = labels.to(device) if labels is not None else None  
        #self.centers = nn.Parameter(self.centers.data.to(device))

        # Extract centers_batch using gather
        centers_batch = torch.gather(self.centers, 0, labels.view(-1, 1).expand(-1, self.feat_dim))

        criterion = nn.MSELoss()
        intra_center_loss = criterion(h, centers_batch)

        dist_matrix = torch.cdist(self.centers, self.centers, p=2)
        mask = torch.eye(self.num_classes, device=self.centers.device).bool()
        dist_matrix = dist_matrix.masked_fill(mask, 0)
        # Penalize small distances (maximize distance)
        inter_center_loss = -dist_matrix.sum() / (self.num_classes * (self.num_classes - 1))

        # Combine the two losses
        center_loss = self.alpha * intra_center_loss + (1 - self.alpha) * inter_center_loss

        return center_loss


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        
        # fc2
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        # fc3
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)  # parallel attention branches for n_classes

    def forward(self, x):
        a = self.attention_a(x) # fc2
        b = self.attention_b(x) # fc3
        A = a.mul(b)    # fc2 . fc3
        A = self.attention_c(A)  # N x n_classes --> parallel attention branches for each class

        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class CLAM_SB(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, use_center_loss=True):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        self.batch_counter = 0

        #MODEL ARCHITECTURE
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]   # fc1
        if dropout:
            fc.append(nn.Dropout(0.25)) 
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)  # TO Attn_Net_Gated for fc2, fc3
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        #########
        self.use_center_loss = use_center_loss

        if use_center_loss:
            self.centers = nn.Parameter(torch.randn(n_classes, size[1]))
            self.centers.requires_grad_(True)
            self.center_loss = CenterLoss(n_classes, size[1], self.centers)
        #########

        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)

        num_samples = min(self.k_sample, A.size(1))  # Use the smaller of k_sample or the actual number of patches

        top_p_ids = torch.topk(A, num_samples)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, num_samples, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(num_samples, device)
        n_targets = self.create_negative_targets(num_samples, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)

        num_samples = min(self.k_sample, A.size(1))  # Use the smaller of k_sample or the actual number of patches

        top_p_ids = torch.topk(A, num_samples)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(num_samples, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, is_final_epochs = False, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        
        # h = data from train_loop_clam  --> Shape  K x D  K: No of patches in the WSL image   D:1024
        A, h = self.attention_net(h)  # NxK 
        # A: Attention coefficient --> attention score for kth patch for ith class

        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N   --> normalized attention score for kth patch for ith class 

        ###############
        if self.use_center_loss and label is not None:
            center_loss = self.center_loss(h, label)
        else:
            center_loss = 0.0
        ################
        
        ######Instance level clustering#################
        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
        
        # h.shape  K x D  K: No of patches in the WSL image   D:512    
        # A.shape  1 x K  K: No of patches in the WSL image       
        M = torch.mm(A, h) # slide level representation aggregated per attention distribution 
        # M.shape  1 x 512 

        logits = self.classifiers(M)    # N paralled independent classifiers    --> 1 x N

        # ????????????????????????????????
        Y_hat = torch.topk(logits, 1, dim = 1)[1]

        Y_prob = F.softmax(logits, dim = 1) #  1 x N
        if instance_eval:
            #results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            #'inst_preds': np.array(all_preds)}
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds), 'center_loss': center_loss}
        else:
            results_dict = {'center_loss': center_loss}
        if return_features:
            results_dict.update({'features': M})
        

        if is_final_epochs:
            ############
            # Extract FEATURE EMBEDDINGS from the output of attention_net

            # Get the indices of the top-k attention coefficients for each instance
            # Ensure k does not exceed the number of patches
            k = min(15, A.size(1))
            topk_indices = torch.topk(A, k, dim=1)[1]

            # Extract the rows of h corresponding to the top-k indices
            patch_feature_embeddings = []

            for i, indices in enumerate(topk_indices[0]):
                # Check that indices are within bounds
                valid_indices = indices[indices < h.shape[0]]
                if valid_indices.numel() > 0:
                    selected_rows = h[valid_indices, :]
                    patch_feature_embeddings.append(selected_rows)

            # Stack the selected rows
            patch_feature_embeddings = torch.stack(patch_feature_embeddings)

            slide_feature_embeddings = M       
            ############
        else:
            patch_feature_embeddings, slide_feature_embeddings = None, None

        return logits, Y_prob, Y_hat, A_raw, results_dict, patch_feature_embeddings, slide_feature_embeddings

