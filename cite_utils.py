# utility funcitons for cite-seq data
import scipy
import numpy as np
import pandas as pd
import scanpy.api as sc


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
        counts = adata.obs['n_counts_0']
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
    
def prot_coverage(adata, prot_key = 'prot', prot_names_key = 'prot_names', groupby = 'batch'):
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
    prot_coverage = pd.DataFrame(index = adata.uns[prot_names_key][:4], columns=levels)
    
    for level in levels:
        proteins = adata[adata.obs[groupby] == level].obsm[prot_key]
        perc = np.round((np.sum((proteins == 0), axis = 0) / proteins.shape[0])[:4], 2)
        prot_coverage[level] = perc
        
    return prot_coverage
    


