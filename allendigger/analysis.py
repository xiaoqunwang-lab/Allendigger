import torch
import numpy as np
import pickle
#import time
from torch_geometric.datasets import Planetoid
import scanpy as sc
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from anndata import AnnData
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from typing import Optional, Union
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
import sklearn
import scipy.sparse as sp
from torch_geometric.nn import GAE
from sklearn import mixture
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import anndata as ad
from sklearn.linear_model import SGDClassifier
from allendigger.accessdata import data_slice, get_data, get_section_structure
from allendigger.preprocess import find_overlap_gene,accuracy, macro_precision, macro_f1, macro_recall, roc_auc_score_multiclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold
from scipy.stats import mannwhitneyu
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def get_spatial_domain(
        section: str,
        section_id: int,
        time: str,
        knn: int = 10,
        lr_rate:float = 1e-3,
        #num_of_cluster: Optional[int] = 5,
        epochs: int = 100,
        seed: int = 1,
        weight_decay: int = 1e-5,
        use_variable_genes: bool = False,
        var_genes: list = [],
        plot_cluster_result: bool = False,
        latent_dim: int = 30,
        n_components: int = 30,
        n_neighbors: int = 30,
        n_pcs: int = 30,
        min_dist: float = 0.5,
        plot_spatial_struct: bool = False,
        annot_level: str = 'level_5',
        verbose: bool = False):
    '''
    Return
    ------
    cluster_label: str

    '''
    if not type(section_id) == int:
        raise TypeError('section id should be an integer')
    time_list = ['E11pt5', 'E13pt5', 'E15pt5', 'E18pt5', 'P4', 'P14', 'P28', 'P56','Adult']
    section_list = ['sagittal', 'horizontal', 'coronal']
    if not time in time_list:
        raise ValueError("inputed time is not avalible, avalible time points are" + '\n' +
                         "'E11pt5','E13pt5','E15pt5','E18pt5','P4','P14','P28','P56','Adult'")

    if not section in section_list:
        raise ValueError("inputed section is not avalible, avalible sections are" + '\n' +
                         "'sagittal','horizontal','coronal'")

    if n_pcs > latent_dim:
        raise ValueError("inputed n_pcs exceed the dimension of latent space," + '\n' +
                         "max n_pcs should below" + n_pcs)
    if plot_spatial_struct:
        if not annot_level:
            raise ValueError("to show spatial structures on UMAP,",+ '\n' + "annotation level should be provided")

    section_dict = {'sagittal': 'z', 'horizontal': 'y', 'coronal': 'x'}

    # readin adata
    adata = get_data(time=time)
    # extract data of inputted section and secion_id
    adata = adata[adata.obs[section_dict[section]] == section_id, :]
    if verbose:
        print('Extracting data for section '+section+" section_id "+ str(section_id)+" from stage "+time+'\n')
    if section == 'sagittal':
        coord = { 'X': adata.obs['x'], 'Y': adata.obs['y']}
        coord = pd.DataFrame(coord)
        coord.index = adata.obs['x'].index
        coord.columns = ['x', 'y']
    elif section == 'horizontal':
        coord = {'X': adata.obs['x'], 'Z': adata.obs['z']}
        coord = pd.DataFrame(coord)
        coord.index = adata.obs['x'].index
        coord.columns = ['x', 'z']
    else:
        adata = adata[adata.obs['z']<=math.ceil(adata.obs['z'].max()/2)]
        coord = {'Y': adata.obs['y'], 'Z': adata.obs['z']}
        coord = pd.DataFrame(coord)
        coord.index = adata.obs['y'].index
        coord.columns = ['y', 'z']
    if verbose:
        print('Building spatial Network'+'\n')
    #build spatial network
    k_cutoff = knn
    KNN_list = []
    neigh = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coord)
    dist, indices = neigh.kneighbors(coord)
    for idx in range(indices.shape[0]):
        df = [np.repeat(idx, indices.shape[1]), indices[idx, :], dist[idx, :]]
        df = pd.DataFrame(df).T
        KNN_list.append(df)
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Voxel1', 'Voxel2', 'Distance']
    SpNet = KNN_df.copy()
    SpNet = SpNet.loc[SpNet['Distance'] > 0,]
    cell_idx = dict(zip(range(coord.shape[0]), np.array(coord.index), ))
    SpNet['Voxel1'] = SpNet['Voxel1'].map(cell_idx)
    SpNet['Voxel2'] = SpNet['Voxel2'].map(cell_idx)
    adata.uns['SpNet'] = SpNet
    G_net = adata.uns['SpNet']
    voxel1_list = []
    for g in range(0,G_net.shape[0]):
        tmp= adata.obs_names.to_list().index(G_net['Voxel1'].to_list()[g])
        voxel1_list.append(tmp)

    voxel2_list = []
    for g in range(0, G_net.shape[0]):
        tmp = adata.obs_names.to_list().index(G_net['Voxel2'].to_list()[g])
        voxel2_list.append(tmp)

    G_net['Voxel1'] = voxel1_list
    G_net['Voxel2'] = voxel2_list

    G = sp.coo_matrix((np.ones(G_net.shape[0]), (G_net['Voxel1'], G_net['Voxel2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])

    edgeList = np.nonzero(G)
    edge_index = torch.LongTensor(np.array([edgeList[0], edgeList[1]]))

    # scale expression data into range [0,1]
    mat = adata.X.astype(np.float32)
    if use_variable_genes:
        if var_genes and len([x for x in var_genes if x in adata.var['gene'].to_list()]) >= 50:
            mat = mat[:, var_genes]
        else:
            raise ValueError("the number of inputted spatial variable genes is too few" + '\n' +
                             len(var_genes))

        # sc.pp.highly_variable_genes(adata, min_mean=0.01, max_mean=2, min_disp=0.25)
        # mat = adata.X[:,adata.var['highly_variable']]
    else:
        mat = adata.X


    min_max_scaler = preprocessing.MinMaxScaler()
    mat = min_max_scaler.fit_transform(mat)
    # generator tensor
    mat = torch.tensor(mat)
    nodes = mat
    if verbose:
        print('Spatial Network construction completed'+'\n')
    # build model
    input_shape = np.shape(mat)[1]
    class GCNEncoder(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super(GCNEncoder,self).__init__()
            self.conv1 = GCNConv(in_channels,2*out_channels)
            self.conv2 = GCNConv(2*out_channels, out_channels)
            self.conv3 = GCNConv(out_channels, 2 * out_channels)
            self.conv4 = GCNConv(2*out_channels,in_channels)
        def forward(self,x,edge_index):
            hidden1 = self.conv1(x,edge_index).relu()
            encoded = self.conv2(hidden1, edge_index)
            hidden2 = self.conv3(encoded,edge_index).relu()
            decoded = self.conv4(hidden2, edge_index)
            return encoded,decoded

    nfeatures = nodes.shape[1]
    if verbose:
        print('Building GAE model'+'\n')
    model = GAE(GCNEncoder(nfeatures, latent_dim))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x = nodes.to(device)
    edge_index = edge_index.to(device)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate, weight_decay=weight_decay)
    if verbose:
        print('Training GAE model'+'\n')
    def train():
        model.train()
        optimizer.zero_grad()
        z,out = model.encode(nodes,edge_index)
        loss = model.recon_loss(out,edge_index)
        loss.backward()
        optimizer.step()
        return float(loss)

    for epoch in range(1,epochs+1):
        loss =train()
        if verbose:
            print(f'Epoch:{epoch + 1},Loss:{loss:.4f}')
    if verbose:
        print('processing clustering analysis'+'\n')
    encoded,decoded = model.encode(nodes,edge_index)
    encoded_np = encoded.detach().numpy()
    adata.obsm['X_GAE'] = encoded_np
    gmm = mixture.GaussianMixture(n_components= n_components)
    gmm.fit(adata.obsm['X_GAE'])
    labels = gmm.predict(adata.obsm['X_GAE'])
    gmm_c = pd.DataFrame({'cluster':labels.astype(str)},index=adata.obs['x'].index)
    adata.obs['gmm'] = gmm_c

    #generate UMAP with latent embeddings

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep='X_GAE')
    sc.tl.umap(adata, min_dist=min_dist)
    if verbose:
        print('whole process completed successfully!')

    # plot cluster result
    if plot_cluster_result:
        fig = sc.pl.umap(adata,color='gmm')
    # plot spatial strcuture
    if plot_spatial_struct:
        fig1 = sc.pl.umap(adata,color = annot_level)
    return adata


# adata = get_spatial_domain('coronal',14,'Adult',n_components=30,lr_rate=1e-4,weight_decay=1e-5,plot_cluster_result=True,plot_spatial_struct=True,annot_level='level_5',verbose=True)


def find_differential_expression_marker(time: str,
                                        labels: list,
                                        anno_level: str,
                                        log2fc_thresh=1.0,
                                        pvalue_thresh=0.05,
                                        cmap='OrRd',
                                        plot_result=True,
                                        method='ranksum',
                                        n=30):
    '''
    find the differentiL expression gene from different brain regions using wilcoxon rank sum test
    :param time: str, the developing period of mouse data ['E11pt5','E13pt5','E15pt5','E18pt5','P4','P14','P28','P56','Adult']
    :param labels: list, the target structure acronyms
    :param anno_level: str, the annotation level from level_1 to level_10
    :param plot_result: bool, whether to plot the heatmap containing auc value of top gene markers
    :param method: str, the method used
    :param n: int, the cutoff of top genes
    :return: dict, the keys represent target structure acronyms, the values are a list of top genes and a DataFrame of auc value of all genes
    '''
    adata = get_data(time=time)
    genes = adata.var.index.tolist()
    result = {}
    for label in labels:
        df = pd.DataFrame(index=adata.var['gene'].tolist(), columns=['auc', 'pvalue','log2FC'])
        x = adata[adata.obs[anno_level] == label]
        y = adata[adata.obs[anno_level] != label]
        FC=[]
        for gene in genes:
            xx = x[:, gene].X
            yy = y[:, gene].X
            mean_x = xx.mean()+1e-5
            mean_y = yy.mean()+1e-5
            fc = (mean_x/mean_y)+1e-5
            FC.append(fc)
        log2FC = np.log2(FC)
        df['log2FC'] = log2FC
        U1, pvalue = mannwhitneyu(x.X, y.X, alternative='two-sided')
        n1 = x.shape[0]
        n2 = y.shape[0]
        df['auc'] = U1/(n1*n2)
        df['pvalue'] = pvalue
        df_l = df[df['log2FC'] > log2fc_thresh]
        df_p = df_l[df_l['pvalue'] < pvalue_thresh]
        df_s = df_p.sort_values(by=['auc'], ascending=False)
        s_max = df_s.iloc[:n-1, :].index.tolist()
        top = list(set(s_max))
        result[label] = [top, df]
    if plot_result:
        columns = []
        for label_ in labels:
            gene = result[label_][0]
            for g in gene:
                columns.append(g)
        # columns = list(set(columns))
        result_df = pd.DataFrame(index=labels, columns=columns)
        for label in labels:
            for column in columns:
                result_df.loc[label, column] = result[label][1].loc[column, 'auc']
        array = result_df.values.astype('float64')
        result_df = pd.DataFrame(array, index=labels, columns=columns)
        plt.figure(figsize=(40,8))
        matplotlib.style.use('default')
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 20
        sns.heatmap(result_df, xticklabels=True, yticklabels=True, linewidths = 0.1,cmap=cmap,annot=False)

        plt.show()

    return result


'''adata = get_data(time='Adult')
labels = list(set(adata.obs['level_8'].tolist()))
result = find_differential_expression_marker(time='Adult', labels=labels, anno_level='level_8')
'''


def section_find_differential_expression_marker(time,
                                                section,
                                                section_id,
                                                labels,
                                                anno_level,
                                                log2fc_thresh=1.0,
                                                pvalue_thresh=0.05,
                                                n=30,
                                                cmap='OrRd',
                                                plot_result=True,
                                                method='ranksum',
                                                ):
    # find the differential expression genes among structures in one section
    i, j, adata = data_slice(time=time, section=section, section_id=section_id)
    genes = adata.var.index.tolist()
    result = {}
    genes = adata.var['gene'].tolist()
    for label in labels:
        df = pd.DataFrame(index=adata.var['gene'].tolist(), columns=['auc', 'pvalue', 'log2FC'])
        x = adata[adata.obs[anno_level] == label]
        y = adata[adata.obs[anno_level] != label]
        FC=[]
        for gene in genes:
            xx = x[:, gene].X
            yy = y[:, gene].X
            mean_x = xx.mean()+1e-5
            mean_y = yy.mean()+1e-5
            fc = (mean_x/mean_y)+1e-5
            FC.append(fc)
        log2FC = np.log2(FC)
        df['log2FC'] = log2FC
        U1, pvalue = mannwhitneyu(x.X, y.X)
        n1 = x.shape[0]
        n2 = y.shape[0]
        df['auc'] = U1/(n1*n2)
        df['pvalue'] = pvalue
        df_l = df[df['log2FC'] > log2fc_thresh]
        df_p = df_l[df_l['pvalue'] < pvalue_thresh]
        df_s = df_p.sort_values(by=['auc'], ascending=False)
        s_max = df_s.iloc[:n-1, :].index.tolist()
        top = list(set(s_max))
        result[label] = [top, df]
    if plot_result:
        columns = []
        for label_ in labels:
            gene = result[label_][0]
            for g in gene:
                columns.append(g)
        result_df = pd.DataFrame(index=labels, columns=columns)
        for label in labels:
            for column in columns:
                result_df.loc[label, column] = result[label][1].loc[column, 'auc']
        array = result_df.values.astype('float64')
        result_df = pd.DataFrame(array, index=labels, columns=columns)
        plt.figure(figsize=(40,8))
        matplotlib.style.use('default')
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 20
        sns.heatmap(result_df, xticklabels=True, yticklabels=True, linewidths = 0.1,cmap=cmap,annot=False)
        plt.show()
    return result



def select_feature(adata,
                  anno_level,
                  n=70):
    genes = adata.var.index.tolist()
    df = pd.DataFrame(adata.X,index=adata.obs.index.tolist(),columns=adata.var.index.tolist())
    x = df
    y = adata.obs[anno_level]
    st_x = preprocessing.StandardScaler()
    x_train = st_x.fit_transform(x)
    sgd = SGDClassifier(max_iter=1000, tol=1e-3,loss='log',penalty='elasticnet',random_state=42)
    sgd.fit(x_train,y)
    classes_sgd = sgd.classes_.tolist()
    coef_sgd_df = pd.DataFrame(sgd.coef_, index=classes_sgd, columns=adata.var.index.tolist())
    gene_list = []
    for class_ in classes_sgd:
        df_ = coef_sgd_df.sort_values(class_,ascending=False,axis=1)
        genes = df_.iloc[:,0:n].columns.tolist()
        gene_list += genes
    gene_list = list(set(gene_list))
    return gene_list


def structure_mapping(adata_spatial,
                      time,
                      anno_level,
                      scrna,
                      true_label=None,
                      evaluate=False,
                      gene_use=None,
                      plot_predict_result=True,
                      method='random forest'):
    '''
    register cells into structures using randomforestclassifier
    :param adata_spatial: AnnData object, voxel-gene expression data 
    :param time: str, the developing period of mouse data ['E11pt5','E13pt5','E15pt5','E18pt5','P4','P14','P28','P56','Adult']
    :param anno_level:str, the annotation level from level_1 to level_10, consistent with the label of scrna data
    :param scrna: AnnData object, cell-gene expression data of scRNA-seq
    :param true_label: str, the obs_name of structure annotation label from scrna data
    :param gene_use: list, the genes used for classifier's training and predicting
    :param plot_predict_result: bool, whether to show the prediction result using confusionmatrixdisplay
    :param method: str, method used for registration
    :return: the classifier, anndata of scrna data with prediction labels
    '''
    if gene_use:
        gene_use = gene_use
    else:
        gene_use = find_overlap_gene(adata_spatial, scrna, time=time)
    genes = scrna.var.index.tolist()
    gene_use = list(set(gene_use)&set(genes))
    # random forest training
    # putting feature variable to X and target variable to y
    df = pd.DataFrame(adata_spatial.X, index=adata_spatial.obs.index.tolist(),columns=adata_spatial.var.index.tolist())
    df = df[gene_use]
    x = df
    y = adata_spatial.obs[anno_level]
    label = y.astype('category')
    labely = label.cat.codes
    x_train = x
    y_train = labely
    st_x = preprocessing.StandardScaler()
    x_train = st_x.fit_transform(x_train)
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    params = {
        'max_depth': [2, 3, 5, 10, 20],
        'min_samples_leaf': [5, 10, 20, 50, 100, 200],
        'n_estimators': [10, 25, 30, 50, 100, 200]
    }
    grid_search = GridSearchCV(estimator=rf,
                               param_grid=params,
                               cv=KFold(4, random_state=2,shuffle=True),
                               n_jobs=-1, verbose=1, scoring="accuracy")
    grid_search.fit(x_train, y_train)
    rf_best = grid_search.best_estimator_
    if time == 'Adult':
        gene_use_ = gene_use
    else:
        gene_use_ = []
        for gene in scrna.var['gene'].tolist():
            if gene.upper() in gene_use:
                gene_use_.append(gene)
    scrna_ = scrna[:, gene_use_]
    scrna_X = st_x.fit_transform(scrna_.X)
    result = rf_best.predict(scrna_X)
    # probablity = rf.predict_proba(scrna_X)
    l = []
    for i in result:
        label_ = label.cat.categories[i]
        l.append(label_)
    label_name = 'pred_rf_' + anno_level
    scrna.obs[label_name] = l

    # evaluate the prediction
    if evaluate:
        if true_label == None:
            raise ValueError("please provide the true_label which is the obs name of the structure label of scrna data")
        else:
            y_true = scrna.obs[true_label]
            y_pred = scrna.obs[label_name]
            cm_display = ConfusionMatrixDisplay.from_predictions(y_true,y_pred,normalize='true',xticks_rotation='vertical')

            print(f"Accuracy: {accuracy(y_true, y_pred)}")
            print(f"Macro-averaged Precision score : {macro_precision(y_true, y_pred)}")
            # print(f"Micro-averaged Precision score : {micro_precision(y_true, y_pred)}")
            print(f"Macro-averaged recall score : {macro_recall(y_true, y_pred)}")
            # print(f"Micro-averaged recall score : {micro_recall(y_true, y_pred)}")
            print(f"Macro-averaged f1 score : {macro_f1(y_true, y_pred)}")
            # print(f"Micro-averaged recall score : {micro_f1(y_true, y_pred)}")
            # roc_auc_dict = roc_auc_score_multiclass(y_true, y_pred)
            
    # plot the predicted result
    if plot_predict_result:
        sc.pl.umap(scrna, color=label_name)
    return rf_best, scrna

