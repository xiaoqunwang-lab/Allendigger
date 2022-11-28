import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from allendigger.accessdata import data_slice, get_data
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.pyplot import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scanpy as sc
import anndata as ad


def plot2D_anno(time, section, section_id, anno_level, cmap='coolwarm',legend=True,label=None, path=None):
    '''
    Plot a slice of annotation data and show.
    :param time: the developing period of mouse data ['E11pt5','E13pt5','E15pt5','E18pt5','P4','P14','P28','P56']
    :param section: the direction of slicing 'sag' for sagittal,'hor' for horizontal,'cor' for coronal
    :param section_id: No. section number
    :param anno_level: str, the annotation level of structure labels
    :param path: None or a path string of a directory to save the image
    :param cmap: str
    :param legend: bool, whether to plot the legend
    :param label: None or list, user can assign the target structures to 'label' list.
    :return: None
    example: plot2D_anno(time='P56', section='Coronal', section_id=20,anno_level='level_3', path='data/figure/')
    '''
    x, y, adata = data_slice(time=time, section=section, section_id=section_id)
    x = adata.obsm['spatial'].iloc[:, 0]
    y = adata.obsm['spatial'].iloc[:, 1]
    if label:
        labels = adata.obs[anno_level]
        labels = labels.astype('str')
        for i in labels.index.tolist():
            if labels.loc[i, ] in label:
                continue
            else:
                labels.loc[i, ] = 'other'
        labely = labels.astype('category')
    else:
        label = adata.obs[anno_level]
        labely = label.astype('category')
    labelx = labely.cat.codes
    num = len(set(labelx))
    c = plt.get_cmap(cmap, num)
    ax = plt.subplot()
    ax.scatter(x, y, s=100, c=c(labelx), marker='s')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axis('equal')
    ax.set_title(time + ' ' + section + ' No.' + str(section_id))
    frame = plt.gca()
    frame.invert_yaxis()
    frame.axes.get_yaxis().set_visible(False)
    frame.axes.get_xaxis().set_visible(False)
    if legend:
        patchlist = []
        for m in range(num):
            label_ = labely.cat.categories[m]
            patch = mpatches.Patch(color=c(m), label=label_)
            patchlist.append(patch)
        ax.legend(handles=patchlist, loc='center left', bbox_to_anchor=(1, 0.5))
    if path:
        plt.savefig(path+time + '_' + section + '_No_' + str(section_id) + '_' + anno_level +'.png' )
    plt.show()


def plot3D_anno(time, anno_level, cmap='coolwarm', legend=True,path=None):
    adata = get_data(time=time)
    size = {'E11pt5': [70, 75, 40],
            'E13pt5': [89, 109, 69],
            'E15pt5': [94, 132, 65],
            'E18pt8': [67, 43, 40],
            'P4': [77, 43, 50],
            'P14': [68, 40, 50],
            'P28': [73, 41, 53],
            'P56': [67, 41, 58],
            'Adult': [67, 41, 58]
    }
    x = adata.obs['x'].tolist()
    y = adata.obs['y'].tolist()
    z = adata.obs['z'].tolist()
    array = np.zeros(size[time])
    size_c = size[time]
    size_c.append(4)
    color = np.zeros(size_c)
    label = adata.obs[anno_level]
    labely = label.astype('category')
    labelx = labely.cat.codes
    num = len(set(labelx))
    c = plt.get_cmap(cmap, num)
    for (i, j, k, l) in zip(x, y, z, labelx):
        array[i][j][k] = 1
        color[i][j][k] = np.asarray(c(l))
    fig = plt.figure()
    ax = plt.subplot(projection='3d')
    ax.voxels(array, facecolors=color, edgecolor=None, alpha=0.3)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(time + ' Mouse ' + anno_level)
    if legend:
        patchlist = []
        for m in range(num):
            label_ = labely.cat.categories[m]
            patch = mpatches.Patch(color=c(m), label=label_)
            patchlist.append(patch)
        ax.legend(handles=patchlist, loc='center left', bbox_to_anchor=(1, 0.5))
    if path:
        plt.savefig(path+time + '_Mouse_' + anno_level + '.png')

    plt.show()



def plot2D_expression(gene_name, time, section, section_id, anno_level, cmap='coolwarm',dcolor='orange',label=None, legend=True, path=None):
    maxx, maxy, adata = data_slice(time=time, section=section, section_id=section_id)
    x = adata.obsm['spatial'].iloc[:, 0]
    y = adata.obsm['spatial'].iloc[:, 1]
    X = adata[:, adata.var['gene'] == gene_name]
    # sc.pp.scale(X)
    if label:
        labels = adata.obs[anno_level]
        labels = labels.astype('str')
        for i in labels.index.tolist():
            if labels.loc[i, ] in label:
                continue
            else:
                labels.loc[i, ] = 'other'
        labely = labels.astype('category')
    else:
        label = adata.obs[anno_level]
        labely = label.astype('category')
    labelx = labely.cat.codes
    num = len(set(labelx))
    c = plt.get_cmap(cmap, num)
    ax = plt.subplot()
    ax.scatter(x, y, s=40, c=c(labelx), marker='s')
    scatter = ax.scatter(x, y, s=X.X, c=dcolor)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axis('equal')
    ax.set_title(time + ' ' + section + ' ' + ' No.' + str(section_id) + ' gene:' + gene_name )
    frame = plt.gca()
    frame.invert_yaxis()
    frame.axes.get_yaxis().set_visible(False)
    frame.axes.get_xaxis().set_visible(False)
    if legend:
        patchlist = []
        for m in range(num):
            label_ = labely.cat.categories[m]
            patch = mpatches.Patch(color=c(m), label=label_)
            patchlist.append(patch)
        legend1 = ax.legend(handles=patchlist, loc='center left', bbox_to_anchor=(1, 0.5),title='Region')
        ax.add_artist(legend1)
        handles, labels = scatter.legend_elements(prop="sizes", color='black', alpha=0.6)
        legend2 = ax.legend(handles, labels, loc="center right", bbox_to_anchor=(0, 0.5), title="Expression")
    if path:
        plt.savefig(path+time + '_' + section + '_' + '_No_' + str(section_id) + '_gene_' + gene_name + '.png')
    plt.show()


    
def plot3D_expression(gene_name, time, anno_level, cmap='coolwarm',dcolor='orange',legend=True,path=None):
    '''
    plot the gene expression distribution on 3D voxel space
    :param gene_name: str, the gene symbol of targeted gene
    :param time: str, the developing period of mouse data ['E11pt5','E13pt5','E15pt5','E18pt5','P4','P14','P28','P56','Adult']
    :param anno_level: str, the annotation level from level_1 to level_10
    :param dcolor: str, the color of dot
    :param legend: bool, whether to plot the legend of structure label and expression
    :return:
    '''
    adata = get_data(time=time)
    size = {'E11pt5': [70, 75, 40],
            'E13pt5': [89, 109, 69],
            'E15pt5': [94, 132, 65],
            'E18pt8': [67, 43, 40],
            'P4': [77, 43, 50],
            'P14': [68, 40, 50],
            'P28': [73, 41, 53],
            'P56': [67, 41, 58],
            'Adult': [67, 41, 58]
            }
    x = adata.obs['x'].tolist()
    y = adata.obs['y'].tolist()
    z = adata.obs['z'].tolist()
    array = np.zeros(size[time])
    size_c = size[time]
    size_c.append(4)
    color = np.zeros(size_c)
    label = adata.obs[anno_level]
    labely = label.astype('category')
    labelx = labely.cat.codes
    num = len(set(labelx))
    c = plt.get_cmap(cmap, num)
    for (i, j, k, l) in zip(x, y, z, labelx):
        array[i][j][k] = 1
        color[i][j][k] = np.asarray(c(l))
    sc.pp.scale(adata)
    data = adata[:, adata.var['gene'] == gene_name]
    s = data.X
    fig = plt.figure()
    ax = plt.subplot(projection='3d')
    ax.voxels(array, facecolors=color, edgecolor=None, alpha=0.3)
    scatter = ax.scatter(x, y, z, s=s, c=dcolor, alpha=0.1)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(time + ' Mouse ' + gene_name +' ' + anno_level)
    if legend:
        patchlist = []
        for m in range(num):
            label_ = labely.cat.categories[m]
            patch = mpatches.Patch(color=c(m), label=label_)
            patchlist.append(patch)
        legend1 = ax.legend(handles=patchlist, loc='center left', bbox_to_anchor=(1, 0.5), title='Region')
        ax.add_artist(legend1)
        handles, labels = scatter.legend_elements(prop="sizes", color='orange', num=5, alpha=0.6)
        legend2 = ax.legend(handles, labels, loc="center right", bbox_to_anchor=(0, 0.5), title="Expression")
    if path:
        plt.savefig(path+time + '_Mouse_' + gene_name +'_' + anno_level+'.png')
    plt.show()


def plot_voxel_distribution(adata,anno_level):
    level = adata.obs[anno_level].tolist()
    labels = set(adata.obs[anno_level].tolist())
    k={}
    for label in labels:
        k[label] = level.count(label)
    names = list(k.keys())
    values = list(k.values())
    plt.figure(figsize=(15,8))
    plt.bar(range(len(k)), values, tick_label=names)
    plt.xticks(rotation = 90)
    plt.show()