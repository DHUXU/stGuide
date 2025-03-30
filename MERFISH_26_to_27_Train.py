import warnings
warnings.filterwarnings("ignore")
import anndata
import scanpy as sc
import random
from stGuide.mnn_utils import *
from stGuide.Utilities import *
from stGuide.Module import *
import torch

used_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(used_device)

# Set seed
seed = 666
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
refer_section_ids = ['data26']
query_section_ids = ['data27']

data_path = '.../'
section_ids = refer_section_ids + query_section_ids
print(section_ids)
section_str = '_'.join(section_ids)
Batch_list = []
adj_list = []

for idx, section_id in enumerate(section_ids):
    print(f'section_id = {section_id} .....')
    # Read h5ad file
    adata = sc.read_h5ad(data_path + f'MERFISH_{section_id}.h5ad')
    adata.var_names_make_unique(join="++")

    adata.obs['batch_name_idx'] = idx

    adata.obs['Ground Truth'] = adata.obs['ground_truth']
    
    if section_id in refer_section_ids:
        adata.obs['identity'] = 'reference'
    else:
        adata.obs['identity'] = 'query'

    # Make spot name unique
    adata.obs_names = [x + '_' + section_id for x in adata.obs_names]

    # Construct intra-edges
    Cal_Spatial_Net(adata, rad_cutoff=150)

    # Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]

    print('Performing PCA dimensionality reduction .....')
    sc.tl.pca(adata, n_comps=10, random_state=seed)

    adj_list.append(adata.uns['adj'])
    Batch_list.append(adata)


# Concat all scanpy objects
adata_concat_all = anndata.concat(Batch_list, label="slice_name", keys=section_ids)
adata_concat_all.obs['Ground Truth'] = adata_concat_all.obs['Ground Truth'].astype('category')
adata_concat_all.obs["batch_name"] = adata_concat_all.obs["slice_name"].astype('category')
# adata_concat_all.obs["batch_name_code"] = adata_concat_all.obs["batch_name_code"].astype('category')
adata_concat_all.obs['identity'] = adata_concat_all.obs['identity'].astype('category')
print('\nShape of concatenated AnnData object (all): ',adata_concat_all.shape)

adj_concat_all = inter_linked_graph(adj_list, section_ids, mnn_dict=None)
adata_concat_all.uns['adj'] = adj_concat_all
adata_concat_all.uns['edgeList'] = np.nonzero(adj_concat_all)
# Concat reference scanpy objects
adata_concat_refer = sc.read_h5ad(f'./pretrain_adata/pretrain_refer_26_to_27.h5ad')
print('\nShape of concatenated AnnData object (reference): ', adata_concat_refer.shape)

adata_all_save, feat_recon_losses, graph_recon_losses, refer_emb_recon_losses = Train_Model(adata_concat_all, adata_concat_refer)

# Clustering
mclust_R(adata_all_save, num_cluster=len(np.unique(adata_all_save.obs[adata_all_save.obs['Ground Truth'] != 'unknown']['Ground Truth'])), used_obsm='model_emb')
adata_all_save = adata_all_save[adata_all_save.obs['Ground Truth'] != 'unknown']

del adata_all_save.uns; del adata_all_save.obsp
adata_all_save.write(f'./train_adata/integrated_adata_26_to_27_all_mclust.h5ad', compression='gzip')





