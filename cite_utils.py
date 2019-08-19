# utility funcitons for cite-seq data
import scipy
import numpy as np
import pandas as pd
import scanpy.api as sc
import scipy.io
import anndata

def import_data(data_p1, data_p2, create = True):
    """
    Utility funciton to import both samples together with proteins.
    
    Parameters
    --------
    data_p1, data_p2: str
        data paths
        
    Output
    --------
    adata: AnnData Object
    """

    if not create:
        path = '../data_update_cite_seq/tec_cite_h5_2019jun18'
        return sc.read_h5ad(path + '/adata.h5') 

    adata_list = []
    for d in [data_p1, data_p2]:
        p_genes = d + 'filtered_feature_bc_matrix/'
        features = pd.read_csv(p_genes + 'features.tsv', delimiter='\t', header=None)[1]
        barcodes = pd.read_csv(p_genes + 'barcodes.tsv', delimiter='\t', header=None)
        matrix = scipy.io.mmread(p_genes + 'matrix.mtx')
        adata = anndata.AnnData(matrix.tocsr()) # compared execution time with anndata.read_mtx and this is faster
        adata.obs.index = features.values.tolist()
        adata.var.index =  barcodes[0].str.slice(stop=-2).values.tolist() # index for proteins
        #adata.var.index =  barcodes[0].values.tolist() # index for proteins
        adata_list.append(adata)                                                                                                                                               


    adata_r1 = adata_list[0].T
    adata_r2 = adata_list[1].T         
    adata_r1.var_names_make_unique()
    adata_r2.var_names_make_unique() 


    # Proteomic Data
    prot_list = []
    for d in [data_p1, data_p2]:
        p_prot = d + 'umi_count/'
        features = pd.read_csv(p_prot + 'features.tsv', delimiter='\t', header=None)[0].values.tolist() 
        features = [f[:f.find('-')] for f in features]
        features[-1] = features[-1] + 'd'
        barcodes = pd.read_csv(p_prot + 'barcodes.tsv', delimiter='\t', header=None)
        matrix = scipy.io.mmread(p_prot + 'matrix.mtx')
        prot = pd.DataFrame(data = matrix.todense(), index = features, columns = barcodes[0].values)
        prot_list.append(prot)

    prot_r1 = prot_list[0].T
    prot_r2 = prot_list[1].T

    protein_names = list(prot_r1.columns)

    # bring the adata object in the right order
    adata_r1 = adata_r1[prot_r1.index]
    adata_r2 = adata_r2[prot_r2.index]

    # combine these again
    adata = adata_r1.concatenate(adata_r2)

    # make names unique
    adata_r1.obs_names_make_unique()
    adata_r2.obs_names_make_unique()

    # combine
    prot = pd.concat((prot_r1, prot_r2), axis=0)

    # add the proteins to the adata object
    adata.obsm['prot'] = prot.values

    # add the protein names
    adata.uns['prot_names'] = protein_names

    # Add some annotations 
    # mitochondrial genes
    mito_genes = [name for name in adata.var_names if name.startswith('mt-')]
    adata.obs['percent_mito'] = np.sum(
        adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
    adata.obs['n_counts_0'] = scipy.sparse.csr_matrix.sum(adata.X, axis = 1) # Number of counts in each cell
    adata.obs['n_genes_0'] = scipy.sparse.csr_matrix.sum(adata.X>0, axis = 1) # Number of genes in each cell
    adata.var['n_cells_0'] =  np.sum(adata.X>0, axis = 0).T # Number of cells where the gene is expressed
    adata.var['n_counts_gene'] = np.sum(adata.X, axis = 0).T # Number of counts of each gene across all cells

    # add some protein annotations
    adata.obs['n_proteins'] = np.sum(adata.obsm['prot'][:, :-1] > 0, axis = 1)
    adata.obs['unmapped'] = adata.obsm['prot'][:, -1]

    #write h5 file
    adata.write('../data_update_cite_seq/tec_cite_h5_2019jun18/adata.h5')

    return adata
    




#%% Normalisation and Scaling

# normalise per cell
def normalise_proteins(adata, prot_key = 'prot', method = 'counts'):
    """
    Normalises the protein counts using centered-log ratio (clr) or to the counts (counts)
    
    Parameters
    --------
    adata : AnnData object
        Annoated data matrix
    prot_key : `str`, optional (default: `"prot"`)
        Key to find the proteins in adata.obsm
    method: `str`, optional (default: `"counts"`)
        Normalisation method to use
    
    """
    # check the input
    if prot_key in adata.obsm.keys(): 
        X = adata.obsm[prot_key]
    else:
        raise ValueError('No field \'{}\' found in adata.obsm'.format(prot_key))
    
    if method not in ['counts', 'clr', 'log1p']: 
        raise ValueError('Select a valid method: "counts", "clr" or "log1p"')

    if method == 'counts':
        counts = np.sum(adata.obsm[prot_key], axis=1)
        X = X / counts[:, None]


    if method == 'clr':
        # add one pseudocount
        X = X + 1
        # compute the geometric mean
        gm = scipy.stats.mstats.gmean(X, axis = 1)
        # normalise with this
        X = np.diag(1/gm) @ X
        # log transform
        X = np.log(X)

    if method == 'log1p':
        # add one pseudocount just to the protein data
        X[:, :-3] += 1 
        # log transform
        X[:, :-3] = np.log(X[:, :-3])
    
    # save in the AnnData object
    adata.obsm[prot_key] = X
    
def scale_proteins(adata, prot_key = 'prot'):
    """
    Scale the protein counts per proteins
    
    Parameters
    --------
    adata : AnnData object
        Annotated data matrix
    prot_key : `str`, optional (default: `"prot"`)
        Key to find the proteins in adata.obsm
    
    """
    # check the input
    if prot_key in adata.obsm.keys(): 
        X = adata.obsm[prot_key]
    else:
        raise ValueError('No field \'{}\' found in adata.obsm'.format(prot_key))
        
    adata_prot = sc.AnnData(X = X)
    sc.pp.scale(adata_prot)
    X_scaled = adata_prot.X
    
    # update adata
    adata.obsm[prot_key] = X_scaled
    
#%% protein quality control
    
def prot_coverage(adata, prot_key = 'prot', prot_names_key = 'prot_names', prot_number=15, groupby = 'batch'):
    """
    Utility function to compute the fraction of zeroes in the protein data
    
    Parameters
    --------
    adata : AnnData object
        Annotated data matrix
    prot_key : `str`, optional (default: `"prot"`)
        Key to find the proteins in adata.obsm
    prot_names_key : `str`, optional (default: `"prot_names"`)
        Key to find the protein names in adata.uns
    groupby : `str`, optional (default: `"batch"`)
    
    Returns
    --------
    prot_coverage : pd.DataFrame
        Dataframe storing percent of zeroes by batch and protein
    """
    
    # get the levels
    levels = adata.obs[groupby].cat.categories
    
    # initialise dataframe
    prot_coverage = pd.DataFrame(index = adata.uns[prot_names_key][:prot_number], columns=levels)
    
    for level in levels:
        proteins = adata[adata.obs[groupby] == level].obsm[prot_key]
        perc = np.round((np.sum((proteins == 0), axis = 0) / proteins.shape[0])[:prot_number], 2)
        prot_coverage[level] = perc
        
    return prot_coverage
    


