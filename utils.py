import os
import matplotlib
matplotlib.use('Agg')
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def infonce_loss(tensorA,tensorB, temperature=10):
    """
    Calculate InfoNCE Loss for each expert in tensorA.
    
    Args:
        tensorA: Tensor of shape (batch_size, expert_num, hidden_dim), where each expert's 3D vector is treated as a positive sample.
        temperature: Temperature scaling factor for softmax.
        tensorB: Tensor of shape (batch_size*node_number,expert_num), where every row is a one_hot vector to represent which expert
        is used for this sample data 
    
    Returns:
        MoCo loss for each expert.
    """
    batch_size, expert_num, hidden_dim = tensorA.shape
    init_tensorA = tensorA.clone()
    if torch.isnan(init_tensorA).any() or torch.isinf(init_tensorA).any():
        print("Input Tensor contains NaN or Inf values.")
    # Normalize the vectors along the last dimension (i.e., the 3D vector for each expert)
    tensorA = F.normalize(tensorA, dim=-1,eps=1e-8)
    if torch.isnan(tensorA).any() or torch.isinf(tensorA).any():
        print("Tensor contains NaN or Inf values.")
    tensorB = (tensorB>0.5) # Get the one-hot vector
    # Create an empty list to store the loss for each expert
    loss_list = []
    
    # For each expert, calculate the positive and negative samples
    for i in range(expert_num):
        # Positive sample for expert i
        positive_sample = tensorA[:, i, :]  # shape: (batch_size, 3)
        
        # Negative samples (all other experts except i)
        
        batch_size = tensorA.shape[0]
        mask = ~torch.eye(batch_size, dtype=bool)
        # Calculate logits: similarity of positive sample with all negative samples
        logits_pos = (positive_sample @ positive_sample.T)/ temperature  # (batch_size,batch_size)
        
        logits_pos = logits_pos[mask].view(batch_size, batch_size - 1) #(batch_size,batch_size-1)
        negative_samples = tensorA[:, torch.arange(expert_num) != i, :]  # shape: (batch_size, expert_num-1, 3)
        negative_samples = negative_samples.reshape(-1,hidden_dim)
        # Calculate logits for all negatives
        logits_neg = torch.matmul(negative_samples, positive_sample.T).T / temperature  # shape: (batch_size, batch_size*(expert_num-1))
        # Combine positive and negative logits
        logits = torch.cat([logits_pos, logits_neg], dim=-1)  # shape: (batch_size, expert_num*batch_size-1)
        
        query_mask = tensorB[:,i].bool().squeeze(-1)
        loss = -torch.log(torch.sum(torch.exp(logits_pos[query_mask]),dim=1)/torch.sum(torch.exp(logits[query_mask]),dim=1))
        if torch.isnan(torch.mean(loss)):
            continue
        loss_list.append(torch.mean(loss))
    # Return the average loss for all experts
    if len(loss_list)==0:
        breakpoint()
    print(f'We have {len(loss_list)} losses')
    return torch.mean(torch.stack(loss_list))

# def infonce_loss(tensorA, temperature=0.1):
#     """
#     Optimized InfoNCE Loss computation for each expert.

#     Args:
#         tensorA: Tensor of shape (batch_size, expert_num, 3).
#         temperature: Temperature scaling factor for softmax.

#     Returns:
#         Averaged InfoNCE loss for all experts.
#     """
#     batch_size, expert_num, _ = tensorA.shape

#     # Normalize the vectors along the last dimension
#     tensorA = F.normalize(tensorA, dim=-1)  # Shape: (batch_size, expert_num, 3)

#     # Compute the similarity matrix for each sample in the batch
#     # Resulting shape: (batch_size, expert_num, expert_num)
#     sim_matrix = torch.matmul(tensorA, tensorA.transpose(1, 2)) / temperature

#     # Reshape the similarity matrix and labels for batch computation
#     logits = sim_matrix.reshape(-1, expert_num)  # Shape: (batch_size * expert_num, expert_num)
#     labels = torch.arange(expert_num, device=tensorA.device).repeat(batch_size)  # Shape: (batch_size * expert_num,)

#     # Compute cross-entropy loss
#     loss = F.cross_entropy(logits, labels)

#     return loss


def create_folders(args):
    try:
        os.makedirs(args.outf)
    except OSError:
        pass

    try:
        os.makedirs(args.outf + '/' + args.exp_name)
    except OSError:
        pass

    try:
        os.makedirs(args.outf + '/' + args.exp_name + '/images_recon')
    except OSError:
        pass

    try:
        os.makedirs(args.outf + '/' + args.exp_name + '/images_gen')
    except OSError:
        pass

def makedir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass

def normalize_res(res, keys=[]):
    for key in keys:
        if key != 'counter':
            res[key] = res[key] / res['counter']
    del res['counter']
    return res

def plot_coords(coords_mu, path, coords_logvar=None):
    if coords_mu is None:
        return 0
    if coords_logvar is not None:
        coords_std = torch.sqrt(torch.exp(coords_logvar))
    else:
        coords_std = torch.zeros(coords_mu.size())
    coords_size = (coords_std ** 2) * 1

    plt.scatter(coords_mu[:, 0], coords_mu[:, 1], alpha=0.6, s=100)


    #plt.errorbar(coords_mu[:, 0], coords_mu[:, 1], xerr=coords_size[:, 0], yerr=coords_size[:, 1], linestyle="None", alpha=0.5)

    plt.savefig(path)
    plt.clf()

def filter_nodes(dataset, n_nodes):
    new_graphs = []
    for i in range(len(dataset.graphs)):
        if len(dataset.graphs[i].nodes) == n_nodes:
            new_graphs.append(dataset.graphs[i])
    dataset.graphs = new_graphs
    dataset.n_nodes = n_nodes
    return dataset

def adjust_learning_rate(optimizer, epoch, lr_0, factor=0.5, epochs_decay=100):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_0 * (factor ** (epoch // epochs_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr