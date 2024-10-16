import torch
from torch import nn
from models.gcl import GCL, E_GCL, E_GCL_vel, GCL_rf_vel
import torch.nn.functional as F
from tqdm import tqdm
import ollama
from utils import infonce_loss
import numpy as np
    
class LLAMA_EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, expert_num, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, recurrent=False, norm_diff=False, tanh=False):
        super(LLAMA_EGNN, self).__init__()

        self.expert_num = expert_num
        self.n_layers = n_layers
        self.hidden_nf = hidden_nf

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        for i in range(0, n_layers):
            for j in range(0,expert_num):
                self.add_module("gcl_%d%d" % (i,j), E_GCL_vel(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, coords_weight=coords_weight, recurrent=recurrent, norm_diff=norm_diff, tanh=tanh))
        self.add_module("gcl_moe", E_GCL_vel(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, coords_weight=coords_weight, recurrent=recurrent, norm_diff=norm_diff, tanh=tanh))
        self.gate_weights_dict = {}
        self.gate_weights_dict_tmp = {}
        self.update_flag = False
        self.debug = False
        self.node_number = 13
        self.batch_size = 100
        self.choices_dict = {'A':0,'B':0,'C':0,'D':0,'E':0}
        self.to(device)
    
    def update_weights(self, x_full, vel_full, edge_attr_full, x_list,batch_idx,state):
        import string
        def get_kth_letter(k):
            if 0 <= k <= 26:
                return string.ascii_uppercase[k]
            else:
                return None
        def letter_position(letter):
            if letter.isupper() and len(letter) == 1:
                return ord(letter) - ord('A')
            else:
                return None
        gate_weights_list = []
        for data_idx in tqdm(range(0,self.batch_size),desc='Update weights in one batch'):
            x = x_full[data_idx*self.node_number:(data_idx+1)*self.node_number]
            vel = vel_full[data_idx*self.node_number:(data_idx+1)*self.node_number]
            edge_attr = edge_attr_full[data_idx*self.node_number:(data_idx+1)*self.node_number]
            init_information = ''
            for idx in range(0,len(x)):
                if edge_attr[idx][0]==1:
                    elec = 'positive'
                else:
                    elec = 'negative'
                init_information += f'The initial position of ball {idx} is {x[idx].cpu().detach().numpy().tolist()}, and its initial velocity is {vel[idx].cpu().detach().numpy().tolist()}, it carries a {elec} charge.'
                # init_information += f'The initial position of ball {idx} is {x[idx].cpu().detach().numpy().tolist()}, and its initial velocity is {vel[idx].cpu().detach().numpy().tolist()}'
            pred_information = ''
            choices = ''
            valid_answers = ['']
            for expert_idx in range(0,self.expert_num):
                pred_sample = x_list[expert_idx][data_idx*self.node_number:(data_idx+1)*self.node_number].cpu().detach().numpy().tolist()
                per_pred = f'The prediction for these balls after 1 seconds are {pred_sample}.'
                pred_information += f'The prediction of agent {expert_idx} is listed: {per_pred}'
                choices+= f'{get_kth_letter(expert_idx)}.Agent{expert_idx} '
                valid_answers+=[get_kth_letter(expert_idx)]
            def get_valid_response(content):
                while True:
                    response = ollama.chat(model='llama3.1', messages=[
                        {
                            'role': 'user',
                            'content': content,
                        },
                    ],
                    options={'temperature':0})
                    answer = response['message']['content'].strip()
                    if answer in valid_answers:
                        return answer

            content = f"You are an intelligent AI assistant for coding, physical simulation, and scentific discovery. Here are {len(x)} balls with different charges, each with a mass of 1kg. {init_information} All values are in SI units, and k is as 1.0 unit in the Coulomb theory like: k*q_1*q_2/(r^2). Please note: The charges on the balls are significant, and Coulomb forces between them result in strong accelerations. We have {self.expert_num} agents to model this dynamic problem. {pred_information} In your perspective, the prediction from which expert might be a correct answer?:{choices}. Respond with only a single letter from these choices. Do not include any explanation or additional text."
            answer = get_valid_response(content)
            self.choices_dict[answer]+=1
            def create_weight_list(k, answer):
                i = letter_position(answer)
                if k <= 1 or i < 0 or i >= k:
                    raise ValueError("invalid k")
                result = [0.2 / (k - 1)] * k 
                result[i] = 0.8  
                return result
            gate_weights = create_weight_list(self.expert_num,answer)
            gate_weights_list.append(gate_weights)
        if state == 'train':
            self.gate_weights_dict[batch_idx] = torch.tensor(gate_weights_list).to('cuda').repeat(1,self.node_number).reshape(-1,self.expert_num).unsqueeze(-1)
        else:
            self.gate_weights_dict_tmp[batch_idx] = torch.tensor(gate_weights_list).to('cuda').repeat(1,self.node_number).reshape(-1,self.expert_num).unsqueeze(-1)

    def forward(self, h, x, edges, vel, edge_attr,batch_idx):
        # gate_values = self.gating_networks[layer_idx](torch.tensor(r1).cuda())
        # gate_weights = F.softmax(gate_values, dim=-1)
        x_init = x.clone()
        h_init = self.embedding(h).clone()
        h_list = []
        x_list = []
        pred_list = []
        for expert_idx in range(0,self.expert_num):
            h = h_init
            for layer_idx in range(0,self.n_layers-1):
                h, x, _ = self._modules["gcl_%d%d" % (layer_idx,expert_idx)](h, edges, x, vel, edge_attr=edge_attr)
            h_list.append(h.clone())
            x_list.append(x.clone())
            h, x, _ = self._modules["gcl_moe"](h, edges, x, vel, edge_attr=edge_attr)
            pred_list.append(x.clone().detach())
        
        if self.update_flag == 'train':
            self.update_weights(x_init, vel, edge_attr, pred_list,batch_idx,state = 'train')
            h_stack = torch.stack(h_list,dim=1)
            h_moe = h_stack*self.gate_weights_dict[batch_idx]
            h_moe = torch.sum(h_moe,dim=1)
            x_stack = torch.stack(x_list,dim=1)
            x_moe = x_stack*self.gate_weights_dict[batch_idx]
            x_moe = torch.sum(x_moe,dim=1)
            h, x, _ = self._modules["gcl_moe"](h_moe, edges, x_moe, vel, edge_attr=edge_attr)
            
        elif self.update_flag == 'tmp':
            self.update_weights(x_init, vel, edge_attr, pred_list,batch_idx,state = 'tmp')
            h_stack = torch.stack(h_list,dim=1)
            h_moe = h_stack*self.gate_weights_dict_tmp[batch_idx]
            h_moe = torch.sum(h_moe,dim=1)
            x_stack = torch.stack(x_list,dim=1)
            x_moe = x_stack*self.gate_weights_dict_tmp[batch_idx]
            x_moe = torch.sum(x_moe,dim=1)
            h, x, _ = self._modules["gcl_moe"](h_moe, edges, x_moe, vel, edge_attr=edge_attr)
        else:
            h_stack = torch.stack(h_list,dim=1)
            h_moe = h_stack*self.gate_weights_dict[batch_idx]
            h_moe = torch.sum(h_moe,dim=1)
            x_stack = torch.stack(x_list,dim=1)
            x_moe = x_stack*self.gate_weights_dict[batch_idx]
            x_moe = torch.sum(x_moe,dim=1)
            h, x, _ = self._modules["gcl_moe"](h_moe, edges, x_moe, vel, edge_attr=edge_attr)
        return x
    
class LLAMA_EGNN_Sparse(LLAMA_EGNN):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, expert_num, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, recurrent=False, norm_diff=False, tanh=False):
        super(LLAMA_EGNN_Sparse, self).__init__(in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, expert_num=expert_num, device=device, act_fn=act_fn, n_layers=n_layers, coords_weight=coords_weight, recurrent=recurrent, norm_diff=norm_diff, tanh=tanh)

    def forward(self, h, x, edges, vel, edge_attr,batch_idx):
        # gate_values = self.gating_networks[layer_idx](torch.tensor(r1).cuda())
        # gate_weights = F.softmax(gate_values, dim=-1)
        x_init = x.clone()
        h_init = self.embedding(h).clone()
        h_list = []
        x_list = []
        pred_list = []
        for expert_idx in range(0,self.expert_num):
            h = h_init
            for layer_idx in range(0,self.n_layers-1):
                h, x, _ = self._modules["gcl_%d%d" % (layer_idx,expert_idx)](h, edges, x, vel, edge_attr=edge_attr)
            h_list.append(h.clone())
            x_list.append(x.clone()) # every x: (batch_size*node_number,3), representing the output from each expert. The length of x list equal to the number of expert.
            h, x, _ = self._modules["gcl_moe"](h, edges, x, vel, edge_attr=edge_attr)
            pred_list.append(x.clone().detach())
        
        if self.update_flag == 'train':
            self.update_weights(x_init, vel, edge_attr, pred_list,batch_idx,state = 'train')
            h_stack = torch.stack(h_list,dim=1)
            h_moe = h_stack*self.gate_weights_dict[batch_idx]
            h_moe = torch.sum(h_moe,dim=1)
            x_stack = torch.stack(x_list,dim=1) #(batch_size*node_number,expert_number,hidden_dim)
            x_moe = x_stack*self.gate_weights_dict[batch_idx]
            x_moe = torch.sum(x_moe,dim=1)
            h, x, _ = self._modules["gcl_moe"](h_moe, edges, x_moe, vel, edge_attr=edge_attr)
            info_loss = infonce_loss(x_stack,self.gate_weights_dict[batch_idx].clone())
            info_loss += infonce_loss(h_stack,self.gate_weights_dict[batch_idx].clone())
        elif self.update_flag == 'tmp':
            self.update_weights(x_init, vel, edge_attr, pred_list,batch_idx,state = 'tmp')
            h_stack = torch.stack(h_list,dim=1)
            h_moe = h_stack*self.gate_weights_dict_tmp[batch_idx]
            h_moe = torch.sum(h_moe,dim=1)
            x_stack = torch.stack(x_list,dim=1)
            x_moe = x_stack*self.gate_weights_dict_tmp[batch_idx]
            x_moe = torch.sum(x_moe,dim=1)
            h, x, _ = self._modules["gcl_moe"](h_moe, edges, x_moe, vel, edge_attr=edge_attr)
            info_loss = infonce_loss(x_stack,self.gate_weights_dict_tmp[batch_idx].clone())
            info_loss += infonce_loss(h_stack,self.gate_weights_dict_tmp[batch_idx].clone())
        else:
            h_stack = torch.stack(h_list,dim=1)
            h_moe = h_stack*self.gate_weights_dict[batch_idx]
            h_moe = torch.sum(h_moe,dim=1)
            x_stack = torch.stack(x_list,dim=1)
            x_moe = x_stack*self.gate_weights_dict[batch_idx]
            x_moe = torch.sum(x_moe,dim=1)
            h, x, _ = self._modules["gcl_moe"](h_moe, edges, x_moe, vel, edge_attr=edge_attr)
            info_loss = infonce_loss(x_stack,self.gate_weights_dict[batch_idx].clone())
            info_loss += infonce_loss(h_stack,self.gate_weights_dict[batch_idx].clone())
        return x,info_loss
    
class LLAMA_EGNN_Sparse_Spring(LLAMA_EGNN_Sparse):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, expert_num, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, recurrent=False, norm_diff=False, tanh=False):
        super(LLAMA_EGNN_Sparse_Spring, self).__init__(in_node_nf, in_edge_nf, hidden_nf, expert_num,  device = device,act_fn=act_fn, n_layers=n_layers, coords_weight=coords_weight, recurrent=recurrent, norm_diff=norm_diff, tanh=tanh)
        self.device = device
    
    def update_weights(self, x_full, vel_full, edge_attr_full, x_list,batch_idx,state):
        import string
        def get_kth_letter(k):
            if 0 <= k <= 26:
                return string.ascii_uppercase[k]
            else:
                return None
        def letter_position(letter):
            if letter.isupper() and len(letter) == 1:
                return ord(letter) - ord('A')
            else:
                return None
        gate_weights_list = []
        for data_idx in tqdm(range(0,self.batch_size),desc='Update weights in one batch'):
            x = x_full[data_idx*self.node_number:(data_idx+1)*self.node_number]
            vel = vel_full[data_idx*self.node_number:(data_idx+1)*self.node_number]
            edge_attr = edge_attr_full[data_idx*self.node_number*(self.node_number-1):(data_idx+1)*self.node_number*(self.node_number-1)]
            init_information = ''
            for idx in range(0,len(x)):
                neighbor = edge_attr[idx*(self.node_number-1):(idx+1)*(self.node_number-1),0].clone()
                neighbor = torch.cat((neighbor[:idx], torch.tensor([0]).to(self.device), neighbor[idx:]))
                position = torch.nonzero(neighbor==1)
                ball_list = str()
                for ball_idx in position:
                    ball_list+= 'ball {}, '.format(ball_idx[0])
                init_information += f'The initial position of ball {idx} is {x[idx].cpu().detach().numpy().tolist()}, and its initial velocity is {vel[idx].cpu().detach().numpy().tolist()}, it connected {ball_list}'
            
            pred_information = ''
            choices = ''
            valid_answers = ['']
            for expert_idx in range(0,self.expert_num):
                pred_information += f'The prediction of agent {expert_idx} is {x_list[expert_idx][data_idx*self.node_number:(data_idx+1)*self.node_number].cpu().detach().numpy().tolist()}'
                choices+= f'{get_kth_letter(expert_idx)}.Agent{expert_idx} '
                valid_answers+=[get_kth_letter(expert_idx)]

            def get_valid_response(content):
                while True:
                    response = ollama.chat(model='llama3.1', messages=[
                        {
                            'role': 'user',
                            'content': content,
                        },
                    ])
                    answer = response['message']['content'].strip()
                    if answer in valid_answers:
                        return answer

            content = f"You are an intelligent AI assistant for coding, physical simulation, and scentific discovery. Here are {len(x)} balls connected by springs, each with a mass of 1kg. {init_information} All values are in SI units, and k is as 1.0 unit in the Hook theory like: k*x, where x is the distence between different body. Please note: The force on the balls are significant, and forces between them result in strong accelerations. We have {self.expert_num} agents to model this dynamic problem. The output of each agent is their prediction of the position of each ball after 1 second. {pred_information}. In your perspective, which prediction might be a correct answer?: {choices}. Respond with only a single letter from these choices. Do not include any explanation or additional text."
            answer = get_valid_response(content)
            def create_weight_list(k, answer):
                i = letter_position(answer)
                if k <= 1 or i < 0 or i >= k:
                    raise ValueError("invalid k")
                result = [0.2 / (k - 1)] * k 
                result[i] = 0.8  
                return result
            gate_weights = create_weight_list(self.expert_num,answer)
            gate_weights_list.append(gate_weights)
        if state == 'train':
            self.gate_weights_dict[batch_idx] = torch.tensor(gate_weights_list).to('cuda').repeat(1,self.node_number).reshape(-1,self.expert_num).unsqueeze(-1)
        else:
            self.gate_weights_dict_tmp[batch_idx] = torch.tensor(gate_weights_list).to('cuda').repeat(1,self.node_number).reshape(-1,self.expert_num).unsqueeze(-1)