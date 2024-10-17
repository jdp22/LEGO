import argparse
import torch
from n_body_system.dataset_nbody import NBodyDataset
from n_body_system.dataset_nbody import NBodyDynamicsDataset as SimulationDataset
from n_body_system.model import LLAMA_EGNN_Sparse
from model.egno import EGNO
import os
from torch import nn, optim
import json
import time
from tqdm import tqdm

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=100, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='n_body_system/logs', metavar='N',
                    help='folder to output vae')
parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=64, metavar='N',
                    help='learning rate')
parser.add_argument('--model', type=str, default='egnn_vel', metavar='N',
                    help='available models: gnn, baseline, linear, linear_vel, se3_transformer, egnn_vel, rf_vel, tfn')
parser.add_argument('--attention', type=int, default=0, metavar='N',
                    help='attention in the ae model')
parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--degree', type=int, default=2, metavar='N',
                    help='degree of the TFN and SE3')
parser.add_argument('--max_training_samples', type=int, default=3000, metavar='N',
                    help='maximum amount of training samples')
parser.add_argument('--dataset', type=str, default="nbody_small", metavar='N',
                    help='nbody_small, nbody')
parser.add_argument('--time_exp', type=int, default=0, metavar='N',
                    help='timing experiment')
parser.add_argument('--weight_decay', type=float, default=1e-12, metavar='N',
                    help='timing experiment')
parser.add_argument('--div', type=float, default=1, metavar='N',
                    help='timing experiment')
parser.add_argument('--norm_diff', type=eval, default=False, metavar='N',
                    help='normalize_diff')
parser.add_argument('--tanh', type=eval, default=False, metavar='N',
                    help='use tanh')
parser.add_argument('--inference', type=bool, default=False, metavar='N',
                    help='whether use the pretrained model')
parser.add_argument('--node_number', type=int, default=5, metavar='N',
                    help='number of node in each system')
parser.add_argument('--save_model', type=bool, default=True, metavar='N',
                    help='whether to save the training model')
parser.add_argument('--num_timesteps', type=int, default=5,
                    help='The number of time steps.')
parser.add_argument('--sparse_coff', type=float, default=1e-1,
                    help='Sparse cofficient used during training')



time_exp_dic = {'time': 0, 'counter': 0}


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


device = torch.device("cuda" if args.cuda else "cpu")
# device = 'cpu'
loss_mse = nn.MSELoss()

print(args)
try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    os.makedirs(args.outf + "/" + args.exp_name)
except OSError:
    pass


def get_velocity_attr(loc, vel, rows, cols):

    diff = loc[cols] - loc[rows]
    norm = torch.norm(diff, p=2, dim=1).unsqueeze(1)
    u = diff/norm
    va, vb = vel[rows] * u, vel[cols] * u
    va, vb = torch.sum(va, dim=1).unsqueeze(1), torch.sum(vb, dim=1).unsqueeze(1)
    return va


def main():
    dataset_train = NBodyDataset(args,partition='train', dataset_name=args.dataset,
                                    max_samples=args.max_training_samples,node_number = args.node_number)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, drop_last=True)

    dataset_val = NBodyDataset(args,partition='val', dataset_name="nbody_small",max_samples=2000,node_number = args.node_number)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False)

    dataset_test = NBodyDataset(args,partition='test', dataset_name="nbody_small",max_samples=2000,node_number = args.node_number)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False)

    if args.model == 'llama_egnn_sparse':
        model = LLAMA_EGNN_Sparse(in_node_nf=1, in_edge_nf=2, hidden_nf=args.nf, expert_num = 5,device=device, n_layers=args.n_layers, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)
        model.node_number = args.node_number
        model.batch_size = args.batch_size
    else:
        raise Exception("Wrong model specified")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    results = {'epochs': [], 'losess': []}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    for epoch in range(0, args.epochs):
        train(model, optimizer, epoch, loader_train,state=False)
        if epoch % 30 == 0:
            val_loss = train(model, optimizer, epoch, loader_val, backprop=False,state = True)
            test_loss = train(model, optimizer, epoch, loader_test, backprop=False,state = True)
            results['epochs'].append(epoch)
            results['losess'].append(test_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_epoch = epoch
                if args.save_model:
                    torch.save(model.state_dict(), '/home/jiangdapeng/egnn/checkpoint/Charge{}_{}_model_weights.pth'.format(args.node_number,args.model))
                    print("Successfully saved model Charge{}_{}_model".format(args.node_number,args.model))
            print("*** Best Val Loss: %.5f \t Best epoch %d" % (best_val_loss, best_epoch))

        json_object = json.dumps(results, indent=4)
        with open(args.outf + "/" + args.exp_name + "/losess.json", "w") as outfile:
            outfile.write(json_object)
    return best_val_loss, best_test_loss, best_epoch

def train(model, optimizer, epoch, loader, backprop=True,state = True):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'coord_reg': 0, 'counter': 0}
    print('length of dataset is:{}'.format(len(loader)))
    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, _ = data[0].size()
        data = [d.to(device) for d in data]
        data = [d.view(-1, d.size(2)) for d in data]
        loc, vel, edge_attr, charges, loc_end = data
        init_loc = loc.clone()
        edges = loader.dataset.get_edges(batch_size, n_nodes)
        edges = [edges[0].to(device), edges[1].to(device)]
        optimizer.zero_grad()
        if args.time_exp:
            torch.cuda.synchronize()
            t1 = time.time()

        if args.model == 'llama_egnn_sparse':
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
            if state:
                model.update_flag = 'tmp'
            elif epoch%200 == 0:
                model.update_flag = 'train'
            else:
                model.update_flag = 'none'
            loc_pred,info_loss= model(nodes, loc.detach(), edges, vel, edge_attr,batch_idx)
        else:
            raise Exception("Wrong model")
        
        a_loss = loss_mse(loc_pred, loc_end)
        if args.model == 'llama_egnn_sparse':
            loss = a_loss+info_loss*args.sparse_coff
            print(f'MSE Loss:{a_loss},info loss:{info_loss}')
        else:
            loss = a_loss
            print(f'batch loss:{a_loss}')
        if backprop:
            loss.backward()
            optimizer.step()
        res['loss'] += loss.item()*batch_size
        res['counter'] += batch_size
        if batch_idx % args.log_interval == 0 and (args.model == "se3_transformer" or args.model == "tfn"):
            print('===> {} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(loader.dataset.partition,
                epoch, batch_idx * batch_size, len(loader.dataset),
                100. * batch_idx / len(loader),
                loss.item()))

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print("[Distance] loss x:%.6f loss y:%.6f loss z:%.6f"% (loss_mse(loc_pred[:,0], loc_end[:,0]),loss_mse(loc_pred[:,1], loc_end[:,1]),loss_mse(loc_pred[:,2], loc_end[:,2])))
    return res['loss'] / res['counter']

if __name__ == "__main__":
    main()




