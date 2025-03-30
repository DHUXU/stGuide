import numpy as np
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
import torch_geometric.transforms as T
from stGuide.gat_conv import GATConv
import torch.optim as optim
from torch_geometric.data import Data
from stGuide.Utilities import *
from torch_geometric.nn import InnerProductDecoder
from torch_geometric.utils import (negative_sampling, remove_self_loops, add_self_loops)
from tqdm.autonotebook import trange
from stGuide.Utilities import EarlyStopping 




class DSBatchNorm(nn.Module):
    """
    Domain-specific Batch Normalization for each slice
    """
    def __init__(self, num_features, n_domains, eps=1e-5, momentum=0.1):
        super().__init__()
        self.n_domains = n_domains
        self.num_features = num_features
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features, eps=eps, momentum=momentum) for i in range(n_domains)])
        
    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()
            
    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()
            
    def _check_input_dim(self, input):
        raise NotImplementedError
            
    def forward(self, x, y):
        out = torch.zeros(x.size(0), self.num_features, device=x.device)
        for i in range(self.n_domains):
            indices = np.where(y.cpu().numpy()==i)[0]

            if len(indices) > 1:
                out[indices] = self.bns[i](x[indices])
            elif len(indices) == 1:
                out[indices] = x[indices]

        return out

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class stGuide_Integration_Model(nn.Module):
    def __init__(self, hidden_dims_integ, mlp_hidden_dim, n_domains, mlp_flag=None, label_num=None, heads=3):
        super(stGuide_Integration_Model, self).__init__()

        # Set network structure
        [in_dim, num_hidden, out_dim] = hidden_dims_integ

        self.fc1 = nn.Linear(in_dim, in_dim)
        self.norm = nn.BatchNorm1d(in_dim)
        self.dropout1 = nn.Dropout(0.2)

        self.conv1 = GATConv(in_dim, num_hidden, heads=heads, concat=False,
                             dropout=0.2, add_self_loops=False, bias=False)

        self.conv_z = GATConv(num_hidden, out_dim, heads=heads, concat=False,
                               dropout=0.2, add_self_loops=False, bias=False)

        if mlp_flag:
            self.mlp = MLP(out_dim, mlp_hidden_dim, label_num)
        
        self.fc2 = nn.Linear(out_dim, in_dim)
        self.dsbnorm = DSBatchNorm(in_dim, n_domains)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(in_dim, in_dim)

        self.graph_decoder = InnerProductDecoder()

    def forward(self, features, edge_index, y, mlp_flag):

        h0 = self.dropout1( F.elu( self.norm( self.fc1(features) ) ) )

        h1 = F.elu(self.conv1(h0, edge_index, attention=True))

        z = self.conv_z(h1, edge_index, attention=True)

        if mlp_flag:
            y_pred = self.mlp(z)
        else:
            y_pred = None

        h2 = self.dropout2( F.elu( self.dsbnorm( self.fc2(z), y ) ) )

        h3 = self.fc3(h2)

        return z, h3, y_pred
    
    def graph_recon_loss(self, z, pos_edge_label_index):

        EPS = 1e-15

        neg_edge_index = None

        reG = self.graph_decoder(z, pos_edge_label_index, sigmoid=True)
        pos_loss = -torch.log(reG + EPS).mean()
        pos_edge_index, _ = remove_self_loops(pos_edge_label_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.graph_decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()
        graph_recon_loss = pos_loss + neg_loss

        return graph_recon_loss
    

    def load_model(self, path):
        """
        Load trained model parameters dictionary.
        Parameters
        ----------
        path
            file path that stores the model parameters
        """
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

def Pretraining_Model(
                        adata, 
                        lr=0.0005,
                        MultiOmics_mode=False, 
                        used_feat='X_seurat',
                        hidden_dims_integ=[512, 10],
                        mlp_dim=64,
                        n_epochs_pre=500, 
                        heads=3,
                        batch_name='batch_name', cell_type='Ground Truth',
                        gradient_clipping=5, weight_decay=0.00001, 
                        key_add='pretrained emb',
                        re_x_weight=0.1, ce_loss_weight = 10,
                        random_seed=666, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                ):

    seed = random_seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # adj_mat = torch.IntTensor(adata.uns['adj']).to(device)
    edgeList = adata.uns['edgeList']

    if MultiOmics_mode:

        data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                    x=torch.FloatTensor(adata.obsm[used_feat]),
                    y=torch.LongTensor([adata.obs[batch_name].cat.codes, adata.obs[cell_type].cat.codes]))

    else:

        data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                    x=torch.FloatTensor(adata.X),
                    y=torch.LongTensor([adata.obs[batch_name].cat.codes, adata.obs[cell_type].cat.codes]))

    # Transform
    transform = T.RandomLinkSplit(num_val=0, num_test=0, is_undirected=True, add_negative_train_samples=False, split_labels=True)
    data, _, _ = transform(data)

    data = data.to(device)

    pretrained_model_path = f'..../pretrain_model_checkpoint.pt'

    early_stopping = EarlyStopping(patience=7, verbose=False, checkpoint_file=pretrained_model_path)
  
    model = stGuide_Integration_Model(hidden_dims_integ=[data.x.shape[1], hidden_dims_integ[0], hidden_dims_integ[1]],
                                        mlp_hidden_dim=mlp_dim,
                                        n_domains=len(adata.obs[batch_name].cat.categories),
                                        heads=heads,
                                        mlp_flag=True,
                                        label_num=len(adata.obs.loc[adata.obs[cell_type]!='unknown', cell_type].unique())).to(device)

    optimizer_stGuide = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    with trange(n_epochs_pre, total=n_epochs_pre, desc='Epochs') as tq:
        for epoch in tq:
            model.train()

            optimizer_stGuide.zero_grad()

            ce = nn.CrossEntropyLoss()

            z, reX, y_pred= model(data.x, data.edge_index, data.y[0], True)

            ce_loss = ce(y_pred, data.y[1])

            features_recon_loss = F.mse_loss(data.x, reX)

            graph_recon_loss = model.graph_recon_loss(z, data.pos_edge_label_index)

            epoch_loss = {'feat_recon_loss':features_recon_loss*re_x_weight, 'graph_recon_loss':graph_recon_loss, 'ce_loss':ce_loss*ce_loss_weight}

            sum(epoch_loss.values()).backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), gradient_clipping)
            optimizer_stGuide.step()

            epoch_info = ','.join(['{}={:.3f}'.format(k, v) for k,v in epoch_loss.items()])
            tq.set_postfix_str(epoch_info) 

            early_stopping(sum(epoch_loss.values()).item(), model)  
            if early_stopping.early_stop:  
                print("Early stopping")
                break

    model.eval()
    if early_stopping.early_stop:  
        model.load_state_dict(torch.load(pretrained_model_path))

    z, _, _ = model(data.x, data.edge_index, data.y, True)
    adata.obsm[key_add] = z.cpu().detach().numpy()
    return adata

def Train_Model(
                adata_all, adata_refer, 
                lr=0.0005, 
                MultiOmics_mode=False, used_feat='X_seurat',
                hidden_dims_integ=[512, 10], 
                n_epochs_pre=1000, 
                heads=3,
                batch_name='identity',
                gradient_clipping=5, 
                weight_decay=0.00001, 
                key_add='model_emb',
                re_x_weight = 1,refer_emb_recon_loss_weight = 10,
                model_save_path='.../',
                random_seed=666, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    
    seed = random_seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # adj_mat = torch.IntTensor(adata_all.uns['adj']).to(device)
    edgeList = adata_all.uns['edgeList']

    if MultiOmics_mode:
        data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                    x=torch.FloatTensor(adata_all.obsm[used_feat]),
                    y=torch.LongTensor(adata_all.obs[batch_name].cat.codes))
    else:
        data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                    x=torch.FloatTensor(adata_all.X),
                    y=torch.LongTensor(adata_all.obs[batch_name].cat.codes))

    # Transform
    transform = T.RandomLinkSplit(num_val=0, num_test=0, is_undirected=True, add_negative_train_samples=False, split_labels=True)
    data, _, _ = transform(data)

    data = data.to(device)

    model = stGuide_Integration_Model(hidden_dims_integ=[data.x.shape[1], hidden_dims_integ[0], hidden_dims_integ[1]],
                                       mlp_hidden_dim=256,
                                       heads=heads,
                                       n_domains=len(adata_all.obs[batch_name].cat.categories),
                                       mlp_flag=False).to(device)

    optimizer_stGuide = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    train_model_path = model_save_path + f'train_adata/train_model/train_model_checkpoint_{refer_emb_recon_loss_weight}.pt'
    
    early_stopping = EarlyStopping(patience=10, verbose=False, checkpoint_file=train_model_path)

    with trange(n_epochs_pre, total=n_epochs_pre, desc='Epochs') as tq:
        for epoch in tq:
            model.train()

            optimizer_stGuide.zero_grad()

            z, reX, y_pred = model(data.x, data.edge_index, data.y, False)

            features_recon_loss = F.mse_loss(data.x, reX)
            refer_emb_recon_loss = F.mse_loss(torch.FloatTensor(adata_refer.obsm['pretrained emb']).to(device), z[adata_all.obs['identity'] == 'reference', :])
            graph_recon_loss = model.graph_recon_loss(z, data.pos_edge_label_index)

            epoch_loss = {'feat_recon_loss': features_recon_loss * re_x_weight, 'graph_recon_loss': graph_recon_loss, 'refer_emb_recon_loss': refer_emb_recon_loss * refer_emb_recon_loss_weight}
            

            sum(epoch_loss.values()).backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), gradient_clipping)
            optimizer_stGuide.step()

            epoch_info = ','.join(['{}={:.3f}'.format(k, v) for k, v in epoch_loss.items()])
            tq.set_postfix_str(epoch_info)

            early_stopping(sum(epoch_loss.values()).item(), model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

    model.eval()

    if early_stopping.early_stop: 
        model.load_state_dict(torch.load(train_model_path))

    z, _, _ = model(data.x, data.edge_index, data.y, False)
    adata_all.obsm[key_add] = z.cpu().detach().numpy()
    return adata_all