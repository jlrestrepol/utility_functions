#!/usr/bin/env python3

from bokeh.palettes import Set1, Set2, Set3
from sklearn.gaussian_process.kernels import *
from pathlib import Path
from collections import Iterable

import anndata
import pandas as pd
import numpy as np
import scanpy.api as sc
import matplotlib.pyplot as plt
import re
import os
import scvelo as scv
import scanpy as sc
import pickle
import traceback

"""
This contains functions for:
* scoring cell cycle genes
* matching DE genes with known marker genes
* plotting marker genes with support for regular expressions
"""

_inter_hist_js_code="""
    // here is where original data is stored
    var x = orig.data['values'];

    x = x.sort((a, b) => a - b);
    var n_bins = bins.value;
    var bin_size = (x[x.length - 1] - x[0]) / n_bins;

    var hist = new Array(n_bins).fill().map((_, i) => { return 0; });
    var l_edges = new Array(n_bins).fill().map((_, i) => { return x[0] + bin_size * i; });
    var r_edges = new Array(n_bins).fill().map((_, i) => { return x[0] + bin_size * (i + 1); });

    // create the histogram
    for (var i = 0; i < x.length; i++) {
        for (var j = 0; j < r_edges.length; j++) {
            if (x[i] <= r_edges[j]) {
                hist[j] += 1;
                break;
            }
        }
    }

    // make it a density
    var sum = hist.reduce((a, b) => a + b, 0);
    var deltas = r_edges.map((c, i) => { return c - l_edges[i]; });
    // just like in numpy
    hist = hist.map((c, i) => { return c / deltas[i] / sum; });

    source.data['hist'] = hist;
    source.data['l_edges'] = l_edges;
    source.data['r_edges'] = r_edges;

    source.change.emit();
"""



class Cache():

    def __init__(self, cache_dir):
        self._cache_dir = Path(cache_dir)
        setattr(self, 'pca', self.cache(dict(obsm='X_pca',
                                             varm='PCs',
                                             uns=['pca', 'variance_ratio'],
                                             uns_cache1=['pca', 'variance']),
                                        default_fname='pca',
                                        default_fn=sc.pp.pca))
        setattr(self, 'neighbors', self.cache(dict(uns='neighbors'),
                default_fname='neighs',
                default_fn=sc.pp.neighbors))
        setattr(self, 'louvain', self.cache(dict(obs='louvain'),
                default_fname='louvain',
                default_fn=sc.tl.louvain))
        setattr(self, 'umap', self.cache(dict(obsm='X_umap'),
                default_fname='umap',
                default_fn=sc.tl.umap))
        setattr(self, 'diffmap', self.cache(dict(obsm='X_diffmap', uns='diffmap_evals', uns_cache1='iroot'),
                default_fname='diffmap',
                default_fn=sc.tl.diffmap))
        setattr(self, 'expression', self.cache(dict(X=None), default_fname='expression'))

    @property
    def cache_dir(self):
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, value):
        if not isinstance(value, Path):
            value = Path(value)

        if not value.is_dir():
            warning.warn(f'`{value}` is not a directory.')

        self._cache_dir = value

    def _create_cache_fn(self, *args, default_fname=None):

        def helper(adata, fname=None, recache=False, verbose=True):

            def _get_val(obj, keys):
                if keys is None:
                    return obj

                if isinstance(keys, str):
                    keys = (keys, )

                for k in keys:
                    obj = obj[k]
                return obj

            try:

                if fname is None:
                    fname = default_fname

                if not fname.endswith('.pkl'):
                    fname += '.pkl'

                if (self.cache_dir / fname).is_file() and not recache:
                    if verbose:
                        print(f'Loading data from: `{fname}`.')
                    with open(self.cache_dir / fname, 'rb') as fin:
                        attrs_keys, vals = zip(*pickle.load(fin))

                    for (attr, key), val in zip(attrs_keys, vals):
                        if key is None or isinstance(key, str):
                            key = (key, )

                        if key[0] is not None:
                            at = getattr(adata, attr)
                            msg = [f'adata.{attr}']

                            for k in key[:-1]:
                                if k not in at.keys():
                                    at[k] = dict()
                                at = at[k]
                                msg.append(f'[{k}]')

                            if verbose and key[-1] in at.keys():
                                print(f'Warning: `{"".join(msg)}` already contains key: `{key[-1]}`.')
                            at[key[-1]] = val
                        else:
                            if verbose and hasattr(adata, attr):
                                print(f'Warning: `adata.{attr}` already exists.')
                            setattr(adata, attr, val)

                else:
                    if verbose:
                        print(f'Caching data to: `{fname}`.')
                    data = [((attr, (key, ) if key is None or isinstance(key, str) else tuple(key)),
                             _get_val(getattr(adata, attr), key)) for attr, key in zip(attrs, keys)]
                    with open(self.cache_dir / fname, 'wb') as fout:
                        pickle.dump(data, fout)

                return True

            except Exception as e:
                if not isinstance(e, FileNotFoundError):
                    if recache:
                        print(traceback.format_exc())
                else:
                    print(f'No cache found; recomputing the values.')
                return False

        if len(args) == 1:
            collection = args[0]
            if isinstance(collection, dict):
                attrs = tuple(collection.keys())
                keys = tuple(collection.values())

                pat = re.compile(r'_cache\d+$')
                attrs = tuple(pat.sub('', a) for a in attrs)

                return helper

            if isinstance(collection, Iterable) and len(next(iter(collection))) == 2:
                attrs, keys = tuple(zip(*collection))

                return helper

        assert len(args) == 2
        attrs, keys = args

        if isinstance(attrs, str):
            attrs = (attrs, )
        if isinstance(keys, str):
            keys = (keys, )

        return helper


    def cache(self, *args, **kwargs):
        """
        Create a caching function.

        :param: keys_attributes (dict, list(tuple))
        :param: default_fname
        """

        default_fn = kwargs.pop('default_fn', None)

        def _run(*args, **kwargs):
            """
            :param: *args
            :param: **kwargs (fname, force, verbose)
            """

            fname = kwargs.pop('fname', None)
            force = kwargs.pop('force', False)
            verbose = kwargs.pop('verbose', True)
            copy = kwargs.get('copy', False)

            callback = None
            if len(args) > 1:
                callback, *args = args

            if len(args) > 0:
                adata = args[0] if isinstance(args[0], anndata.AnnData) else kwargs.get('adata')
            else:
                adata = kwargs.get('adata')

            assert isinstance(adata, anndata.AnnData), f'Expected `{adata}` to be of type `anndata.AnnData`, got `{type(adata)}`.'

            if callback is None:
                callback = (lambda *_x, **_y: None) if default_fn is None else default_fn

            assert callable(callback), f'`{callblack}` is not callable.'

            if force:
                res = callback(*args, **kwargs)
                cache_fn(res if copy else adata, fname, True, verbose=verbose)
                return res

            if not cache_fn(adata, fname, False, verbose):
                print('Recomputing values.')
                res = callback(*args, **kwargs)
                cache_fn(res if copy else adata, fname, True, verbose=False)
                return res

        cache_fn = self._create_cache_fn(*args, **kwargs)

        return _run


def score_cell_cycle(adata, path, gene_symbols = 'none'):
    """
    Computes cell cycle scores. This is usually done on batch corrected data.
        adata - anndata object
        path - path to a file containing cell cycle genes
        gene_symbols - annotation key from adata.var
    """
    
    # import the gene file
    cc_genes = pd.read_table(path, delimiter='\t')
    
    # sort by s and g2m genes
    s_genes = cc_genes['S'].dropna() # s phase genes
    g2m_genes = cc_genes['G2.M'].dropna()  # g2 phase genes

    # change the capitalisasion, to get from human genes to mouse genes
    s_genes_mm = [gene.lower().capitalize() for gene in s_genes]
    g2m_genes_mm = [gene.lower().capitalize() for gene in g2m_genes]

    # which of those are also in our set of genes? in1d is a good way to check wether something
    # is contained in something else
    if gene_symbols is not None:
        s_genes_mm_ens = adata.var_names[np.in1d(adata.var[gene_symbols], s_genes_mm)]
        g2m_genes_mm_ens = adata.var_names[np.in1d(adata.var[gene_symbols], g2m_genes_mm)]
    else:
        s_genes_mm_ens = adata.var_names[np.in1d(adata.var_names, s_genes_mm)]
        g2m_genes_mm_ens = adata.var_names[np.in1d(adata.var_names, g2m_genes_mm)] 

    # call the scoring function
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes_mm_ens,
                                                g2m_genes=g2m_genes_mm_ens)
def check_markers(de_genes, marker_genes):
    """ 
    This function compares a set of marker genes obtained from a differential expression test
    to a set of reference marker genes provided by a data base or your local biologist

    Parameters
    --------
    de_genes: pd.dataFrame
        A data frame of differentially expressed genes per cluster
    marker_genes: dict
        A dict of known marker genes, e.g. for a specific cell type, each with a key

    Output
    --------
    matches: dict
    Genes from the DE list which were found in the marker_genes dict, for a specific key
    """
    # create a dict for the results
    matches = dict()
    
    # loop over clusters
    for group in de_genes.columns:
        
        # add a new entry in the results dict
        matches[group] = dict()
        
        # extract the de genes for that cluster
        de_genes_group = de_genes[group].values
        
        # loop over cell types
        for key in marker_genes:
            
            genes_found = list()
            # loop over the markers for this key
            for gene in marker_genes[key]:
                regex = re.compile('^' + gene + '$', re.IGNORECASE)
                result = [l for l in de_genes_group for m in [regex.search(l)] if m]
                if result: genes_found.append(result[0])
            
            # save the matches in the dict
            if genes_found: matches[group][key] = genes_found
            
    return(matches)


def plot_markers(adata, key, markers = None, basis = 'umap', n_max = 10,
                   use_raw = True, multi_line = True, ignore_case = True,
                   protein= False, min_cutoff = None, max_cutoff = None, 
                   colorbar = False, prot_key = 'prot', prot_names_key = 'prot_names', **kwags):
    """
    This function plots a gridspec which visualises marker genes and a clustering in a given embedding.
    
    Parameters
    --------
    adata: AnnData Object
        Must contain the basis in adata.obsm
    key: str
        Can be either of var annotation, reg. expression or key from the markers dict (if given)
    markers: dict or None
        containing keys like cell types and marker genes as values
    basis: str
        any embedding from adata.obsm is valid
    n_max: int
        max number of genes to plot
    use_raw: boolean
        use adata.raw
    multi_line: boolean
        plot a grid
    protein: boolean
        Indicates wether this key should be interpreted as a protein name. Relevant
        for cite-seq data.
    min_cutoff, max_cutoff: str
        string to indicate quantiles used for cutoffs, e.g. q05 for the 5% quantile
    colorbar: boolean
        wether a colorbar shall be plotted
    prot_key : `str`, optional (default: `"prot"`)
        Key to the proteins in adata.obsm
    prot_names_key : `str`, optional (default: `"prot_names"`)
        Key to the protein names in adata.uns
    **kwags: keywod arguments for plt.scatter
    """
    
    # check wether this basis exists
    if 'X_' + basis not in adata.obsm.keys():
        raise ValueError('You have not computed the basis ' + basis + ' yet. ')
        
    X_em = adata.obsm['X_' + basis]
    if basis == 'diffmap': X_em = X_em[:, 1:]
    
    # give some feedback
    print('Current key: {}'.format(key))
    print('Basis: {}'.format(basis))
    
    # get the gene names
    if use_raw: 
        try:
            print('Using the rawdata')
            var_names = adata.raw.var_names
        except: 
            var_names = adata.var_names
            use_raw = False
            print('adata.raw does not seem to exist')
    else: 
        var_names = adata.var_names
        
    # obtain the subset of genes we would like to plot
    if markers is not None and protein is False:
        
        if key not in markers.keys():
            
            print('Key not in the markers dict. Searching in the var names.')
            if ignore_case:
                reg_ex = re.compile(key, re.IGNORECASE)
            else:
                reg_ex = re.compile(key, re.IGNORECASE)
            genes = [l for l in var_names \
                     for m in [reg_ex.search(l)] if m]
                
        else:
            
            print('Key found in the markers dict.')
            genes_pre = markers[key]
            genes = list()
            not_found = list()
            
            # search through the list of genes
            for gene in genes_pre:
                if ignore_case:
                    reg_ex = re.compile('^' + gene + '$', re.IGNORECASE)
                else: 
                    reg_ex = re.compile('^' + gene + '$')
                result = [l for l in var_names \
                          for m in [reg_ex.search(l)] if m]
                if len(result)> 0:
                    genes.append(result[0])
                else:
                    not_found.append(gene)
            if len(not_found)> 0:
                print('Could not find the following genes: ' + str(not_found))
                
    elif protein is False:
        print('No markers dict given. Searching in the var names.')
        genes = []
        for gene in key:
            if ignore_case:
                reg_ex = re.compile(gene, re.IGNORECASE)
            else:
                reg_ex = re.compile(gene)
            genes_ = [l for l in var_names \
                     for m in [reg_ex.search(l)] if m]
            genes.append(*genes_)
    elif protein is True:
        # we will internally refer to the proteins as genes 
        print('Looking for a protein with this name.')
        
        if (prot_names_key not in adata.uns.keys()) or (prot_key not in adata.obsm.keys()):
            raise ValueError('Requires a filed \'{}\' in adata.uns and a field \'{}\' in adata.obsm'.format(prot_names_key, prot_key))
        proteins = adata.obsm[prot_key]
        protein_names = adata.uns[prot_names_key]
        
        # combine to a dataframe
        proteins = pd.DataFrame(data = proteins, columns=protein_names)
        if ignore_case:
            reg_ex = re.compile(key, re.IGNORECASE)
        else:
            reg_ex = re.compile(key)
        genes = [l for l in protein_names \
                 for m in [reg_ex.search(l)] if m]
        
        
        
    if len(genes) == 0:
        raise ValueError('Could not find any gene or protein to plot.')
        
    # make sure it is not too many genes
    if len(genes) > n_max: 
        print('Found ' + str(len(genes)) + ' matches.')
        genes = genes[:n_max]
    if not protein: 
        print('Plotting the following genes:' + str(genes)) 
    else:
        print('Plotting the following proteins:' + str(genes)) 
            
    # create a gridspec
    n_genes = len(genes)
    
    if multi_line:
        n_col = 3
        n_row = int(np.ceil(n_genes+1/n_col))
    else:
        n_col = n_genes + 1
        n_row = 1
        
    gs = plt.GridSpec(n_row, n_col, figure = plt.figure(None, (12, n_row*12/(n_col+1) ), dpi = 150))
    
    
    # plot the genes
    plt.title(key)
    
    for i in range(n_genes+ 2): 
        plt.subplot(gs[i])
        
        # genes
        if i < n_genes:
            # get the color vector for this gene
            if not protein:
                if use_raw:
                    color = adata.raw[:, genes[i]].X
                else:
                    color = adata[:, genes[i]].X
                plt.title('Gene: ' + genes[i])
            else:
                color = proteins[genes[i]]
                plt.title('Protein: ' + genes[i])
                
            # quantile normalisation
            if min_cutoff is not None:
                color_min = np.quantile(color, np.float(min_cutoff[1:])/100)
            else:
                color_min = np.min(color)
            if max_cutoff is not None:
                color_max = np.quantile(color, np.float(max_cutoff[1:])/100)
            else:
                color_max = np.max(color)
            color = np.clip(color, color_min, color_max)
            
            plt.scatter(X_em[:, 0], X_em[:, 1], marker = '.', c = color, **kwags)
            
            # add a colorbar
            if colorbar: plt.colorbar()
        elif i == n_genes: #louvain
            ax = sc.pl.scatter(adata, basis = basis, color = 'louvain', 
                               show = False, ax = plt.subplot(gs[i]), 
                               legend_loc = 'right margin') 
        elif i > n_genes: #condition
            if 'color' in adata.obs.keys():
                print('found key')
                ax = sc.pl.scatter(adata, basis = basis, color = 'color', 
                                   show = False, ax = plt.subplot(gs[i]),
                                   legend_loc = 'right margin') 
        plt.axis("off")
    plt.plot()
    
def map_to_mgi(adata, copy = False):
    """Utility funciton which maps gene names from ensembl names to mgi names. 
    Queries the biomart servers for the mapping
    
    Parameters
    --------
    adata: AnnData object
    """
    from pybiomart import Server
    # connest to the biomart server
    server = Server(host='http://www.ensembl.org')
    
    # retrieve the mouse data set we need
    dataset = (server.marts['ENSEMBL_MART_ENSEMBL']
                 .datasets['mmusculus_gene_ensembl'])

    # recieve the mapping from ensembl to MGI
    conv_table = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name'])
    
    # we first drop duplicates in the first column
    conv_table = conv_table.drop_duplicates(conv_table.columns.values[0])
    
    # convert the gene names from the adata object to a data frame
    adata_table = pd.DataFrame(adata.var_names)
    
    # give the first column a name
    adata_table.columns = ['Gene stable ID']
    
    # change the gene table so that the ensembl names are now the index
    conv_table = conv_table.set_index('Gene stable ID')
    
    # project the names from the conversion table on the corr. names in the 
    # adata var names table
    mapping = adata_table.join(conv_table, on='Gene stable ID')
    
    # how many could we not map
    not_found_mgi = sum(pd.isnull(mapping).iloc[:,1])

    # how many ensg symbols did we map several times?
    rep_ensg = len(mapping.iloc[:, 0]) - len(set(mapping.iloc[:, 0]))
    
    # how many mgi symbols did we map several times?
    rep_mgi = len(mapping.iloc[:, 1]) - len(set(mapping.iloc[:, 1]))
    
    # print this information
    print('Genes where no MGI annotations where found: {}\nENSG repetition: {}\nMGI repetition: {}'.\
         format(not_found_mgi, rep_ensg, rep_mgi))
    
    # fill nans in mgi column with corresponding ensembl annotations
    mapping['Gene name'].fillna(mapping['Gene stable ID'], inplace = True)
    
    # add the new gene names to the adata object
    adata.var['mgi_symbols'] = mapping['Gene name'].tolist()
    
    
def compare_distr(adata, key, groupby = 'batch', **kwags):
    """
    Utility function that lets you compare quality measures among batches.
    
    Parameters:
    --------
    adata : :class: '~anndata.AnnData`
        Annotated data matrix
    key : `str`
        Observation annotation to use for comparing
    groupby: `str`, optional (default: `"batch"`)
        Levels used for grouping
    **kwags: dict
        Keyword arguments for plt.hist()
        
    Returns:
    --------
    Nothing but it produces nice plots
    """
    
    plt.figure(None, (8, 6), 70)
    levels = adata.obs[groupby].cat.categories
    for level in levels:
        plt.hist(adata[adata.obs[groupby] == level].obs[key], alpha = 0.5, 
                     label = level, density = True , **kwags)
    plt.legend()
    plt.title(key)
    plt.show()
    
    
def print_numbers(adata, groupby=None, return_numbers=False, save_numbers=None):
    """
    Utility function to print cell numbers per batch
    
    Useful when filtering to check at intermediate steps how many cells and genes are left
    
    Parameters:
    --------
    adata : AnnData object
        Annotated data matrix
    groupby : `str`, optional (defalut: `None`)
        Key to categorical annotation in adata.obs
    return_numbers: bool, optional (default: `False`)
        Whether to return cell and gene numbers
    save_numbers: None or str, optional (default: `None`)
        if not None, saved the numbers in the adata object in uns
    """

    adata_numbers = dict()
    adata_numbers['groupby'] = groupby
    adata_numbers['n_cells_total'] = adata.n_obs

    # check wether batch key exists
    if groupby is not None:
        if groupby not in adata.obs.keys():
            raise ValueError('Cannot find the key {!r} in adata.obs'.format(groupby))
        else:
            adata_numbers['by_group'] = dict()
            # get the levels
            groups = adata.obs[groupby]
            groups = pd.Series(groups, dtype='category')
            levels = groups.cat.categories

            # print number of cell per batch
            for level in levels:
                n_cells = adata[adata.obs[groupby] == level].n_obs
                print('{} cells in {} {}'.format(n_cells, groupby, level))
                adata_numbers['by_group'][level] = n_cells
    
    # In total
    print('Total: {} cells, {} genes'.\
          format(adata.n_obs, adata.n_vars))

    if save_numbers is not None:
        adata.uns[save_numbers] = adata_numbers

    if return_numbers:
        return adata_numbers


def print_filtering(adata, key='original_numbers'):
    """

    :param adata: AnnData Object
        Annotated Data Matrix
    :param key: str, optional (default: `"original_numbers"`)
        Where to find the original cell numbers in the
        adata.uns field
    :return: Nothing, just prints
    """
    # check whether the original number have been saved
    if key not in adata.uns.keys():
        raise ValueError('Key {!r} not found in adata.uns'.format(key))
    else:
        original_numbers = adata.uns[key]

    # print results for total numbers
    print('In total: ')
    n_cells = adata.n_obs
    n_cells_orig = original_numbers['n_cells_total']
    print('Filtered out {} cells or {:2.2f} %.\n'.format(
        n_cells_orig - n_cells,
        100 * (n_cells_orig - n_cells) / n_cells_orig))

    # print results for group specific numbers
    group_key = original_numbers['groupby']
    if group_key is not None:
        print('By {}'.format(group_key))
        groups = adata.obs[group_key]
        groups = pd.Series(groups, dtype='category')
        groups = groups.cat.categories
        for group in groups:
            n_cells = adata[adata.obs[group_key] == group].n_obs
            n_cells_orig = original_numbers['by_group'][group]
            print('Filtered out {} cells or {:2.2f} % of {} {}.'.format(
                n_cells_orig - n_cells,
                100 * (n_cells_orig - n_cells) / n_cells_orig,
                group_key,
                group))

def batch_quantification(adata, scale=False,
                         regress_out=None,
                         n_neighbors=15, n_pcs=10,
                         keys=['n_counts', 'n_genes'],
                         batch_quant_key='batch',
                         bases=['pca', 'umap'],
                         components=[1, 2],
                         random_state=0, copy=True):

    """
    adata : AnnData object
        Annotated data matrix
    scale: bool (default: `False`)
        Scale cells to have 0 mean and 1 variance
    regress_out: list[str], optional (default: `None`)
        Keys from adata.obs to regress out
    n_neighbors : int (default: `15`)
        Number of neighbors to consider during clustering
    n_pcs : int (default: `10`)
        Number of principal components to use for clustering
    keys : list[str], optional (default: `['n_counts', 'n_genes']`
        Keys which are used in corr_ann
    batch_quant_key: str, optional (default: `'batch'`)
        Labels to use to compute silhouette coefficient
    bases : list[str], optional (default: `['pca']`)
        Bases to use
    components : list[int], optional (default: `[1, 2]`)
        Components of bases which are being passed to corr_ann
    random_state : int, optional (default: `0`)
        Random state to use
    copy : bool, optional (default: `False`)
        Operate on adata's copy rather than in-place
    Returns
    -------
    None
    """

    if copy:
        adata = adata.copy()

    if regress_out is not None:
        sc.pp.regress_out(adata, regress_out)

    if scale:
        sc.pp.scale(adata)

    sc.pp.pca(adata, svd_solver='arpack', random_state=random_state)

    sc.pp.neighbors(adata, n_neighbors=n_neighbors,
                    n_pcs=n_pcs,
                    random_state=random_state)
    sc.tl.umap(adata, random_state=random_state)

    for basis in bases:
        corr_ann(adata, obs_keys=keys, basis=basis, components=components)
        quant_batch(adata, key=batch_quant_key, basis=basis, components=components)

    return adata if copy else None


def create_dir(dir_type, base_path):
    """Utility function that manages scanpy directories

    :param dir_type: str
        Usually 'write', 'data' or 'figures'
    :param base_path: str
        Base path that is joined with the dir_type
    :return: str
        The joined path
    """

    path = os.path.join(base_path, dir_type)
    if not os.path.exists(path):
        os.mkdir(path)
        print('Created directory {!r}'.format(path))
    else:
        print('Found directory {!r}'.format(path))


    if dir_type.find('figure') != -1:
        sc.settings.figdir = path
        scv.settings.figdir = path

    return path

    
def corr_ann(adata, obs_keys=['n_counts', 'n_genes'], basis='pca', components=[1, 2]):
    """
    Utility function to correlate continuous annotations against embedding
    
    Can be used to see how large the linear influence of a measure like the count depth is on a 
    given component of any embedding, like PCA
    
    Parameters
    --------
    adata : AnnData object
        Annotated data matrix
    obs_keys : `str`, optional (default: `[n_counts, n_genes]`)
        Key for the continious annotation to use
    basis : `str`, optional (default: `"pca"`)
        Key to the basis stored in adata.obsm
    components : `int`, optional (default: `[1, 2]`)
        Component of the embedding to use
        
    
    Returns
    --------
    Nothing, but prints the correlation
    """
    
    # check input
    if 'X_' + basis not in adata.obsm.keys():
        raise ValueError('You have not computed this basis yet')
    for key in obs_keys:
        if key not in adata.obs.keys():
            raise ValueError('The key {!r} does not exist in adata.obs'.format(obs_key))
        
    # get the embedding coordinate
    X_em = adata.obsm['X_' + basis]

    for key in obs_keys:
        for comp in components:
            coordinates = X_em[:, comp - 1]

            # get the continious annotation
            ann = adata.obs[key]

            # compute the correlation coefficient
            corr = np.corrcoef(coordinates, ann)[0, 1]
            print('Correlation between {!r} and component {!r} of basis {!r} is {:.2f}.'.format(key,
                comp, basis, corr))


# quantify the batch effect quickly using the silhouette coefficient
def quant_batch(adata, key = 'batch', basis = 'pca', components=[1, 2]):
    """
    Utility funciton to quantify batch effects
    
    This is just a very simple approach, kBET by Maren will be much better and more sensitive 
    at fulfilling the same task.
    
    Parameters
    --------
    adata : AnnData object
        Annotated data matrix
    key : `str`, optional (default: `"batch"`)
        Labels to use to compute silhouette coefficient
    basis : `str`, optional (default: `"pca"`)
        Basis to compute the silhouette coefficient in. First two components used.
    components : list[str], optional (default: `[1, 2]`)
        Which components to use
        
    Returns
    --------
    Nothing, prints the silhouette coefficient.
    """
    
    from sklearn.metrics import silhouette_score

    # check input
    if 'X_' + basis not in adata.obsm.keys():
        raise ValueError('You have not computd this basis yet')
    if key not in adata.obs.keys():
        raise ValueError('The key \'{}\' does not exist in adata.obs'.format(key))
        
    if not isinstance(components, type(np.array)):
        components = np.array(components)
    
    # get the embedding coordinate
    X_em = adata.obsm['X_' + basis]
    X_em = X_em[:, components - 1]
    
    # get the continious annotation
    ann = adata.obs[key]
    
    # compute silhouette coefficient
    score = silhouette_score(X_em, ann)
    print('Silhouette coefficient in basis \'{}\' for the labels given by \'{}\' is {:.2f}'.format(basis, key, score))


# what's the distribution of the two batches within each cluster?
def cluster_distr(adata, cluster_key = 'louvain', batch_key = 'batch', eps = 0.4):
    """
    Utility function to compute how many cells from each batch are in each cluster.
    
    The aim here is to have a very simple procedure to find clusters which are heavily dominated
    by just one batch, which can be an indication that this cluster is not biologically relevant, but just a
    technical artefact.
    
    Parameters
    --------
    adata : AnnData object
        Annotated data matrix
    cluster_key : `str`, optional (defaul: `"louvain"`)
        Key from adata.obs for the clustering
    batch_key: `str`, optional (default: `"batch"`)
        Key from adata.obs for the batches
    eps : float, optional (default: `0.4`)
        Raises a warning if the entropy for any cluster is smaller than this threshold.
        Can be an indicator strong batch effect in that cluster
        
    Returns
    --------
    batch_distr : pd.DataFrame
        Stores total cells numbers per cluster as well as percentages corresponding to batches.
    """
    
    from scipy.stats import entropy
    
    # check the input
    if cluster_key not in adata.obs.keys():
        raise ValueError('The key \'{}\' does not exist in adata.obs'.format(cluster_key))
    if batch_key not in adata.obs.keys():
        raise ValueError('The key \'{}\' does not exist in adata.obs'.format(batch_key))

    # get the clusters and batches
    clusters = adata.obs[cluster_key].cat.categories
    batches = adata.obs[batch_key].cat.categories

    # initialise dataframe
    batch_distr = pd.DataFrame(index = clusters, columns= 'perc_' + batches)

    # how many cells are there in total per cluster
    cells_per_cluster = [np.sum(adata.obs[cluster_key] == cluster) for cluster in clusters]
    batch_distr['total number'] = cells_per_cluster

    # loop over the batches
    for batch in batches:
        assignment = adata[adata.obs[batch_key] == batch].obs[cluster_key]
        cells_per_cluster_batch = [np.sum(assignment == cluster) for cluster in clusters]
        perc = np.round(np.array(cells_per_cluster_batch) / \
             np.array(cells_per_cluster), 2)
        batch_distr['perc_' + batch] = perc
    
    # compute the entropy
    en  = []
    for cluster in clusters:
        data = batch_distr.loc[cluster][list('perc_' + batches)]
        entropy_cluster = entropy(data)
        en.append(entropy_cluster)
        
        # warn if very small entropy
        if entropy_cluster <= eps:
            print('Warning: Cluster {} has a very uneven batch assignment.'.format(cluster))
    batch_distr['entropy'] = np.round(en, 2)
    
    
    return batch_distr


def de_results(adata, keys = ['names', 'scores'], cluster_key = 'louvain', n_genes = 50):
    """
    Utility function which returns the results of the differential expression test.
    
    Parameters
    --------
    adata: AnnData object
        Annoated data matrix
    keys : list, optional (default: `['names', 'scores']`)
        Columns to be included in the table
    cluster_key : str, optional (default: `"louvain"`)
        Key from adata.obs where cluster assignment is stored
    n_genes : int, optional (default: `50`)
        Number of genes to include in the table
        
    Returns
    --------
    table : pd.DataFrame
        Contains the results of the differential expressin test
    """
    
    # check input
    if cluster_key not in adata.obs.keys():
        raise ValueError('Could not find the key \'{}\' in adata.obs'.format(cluster_key))
    if 'rank_genes_groups' not in adata.uns.keys():
        raise ValueError('Run the differential expression test first.')
        
    # get the dict
    result = adata.uns['rank_genes_groups']
    group_names = result['names'].dtype.names
    
    # construct a lovely table with a dict comprehension
    table = {group + '_' + key[:10]: \
        result[key][group] for group in group_names for key in keys}
    table = pd.DataFrame(table).head(n_genes)
    
    return table


def interactive_histograms(adata, keys=['n_counts', 'n_genes'],
                           bins=100, min_bins=1, max_bins=1000,
                           tools='pan, reset, wheel_zoom, save',
                           groups=None, fill_alpha=0.4,
                           palette=Set1[9] + Set2[8] + Set3[12],
                           legend_loc='top_right', display_all=True,
                           *args, **kwargs):
    """Utility function to plot count distributions\

    Uses the bokey library to create interactive histograms, which can be used
    e.g. to set filtering thresholds.

    Params
    --------
    adata: AnnData Object
        annotated data object
    keys: list, optional (default: `["n_counts", "n_genes"]`)
        keys in adata.obs or adata.var where the distibutions are stored
    bins: int, optional (default: `100`)
        number of bins used for plotting
    min_bins: int, optional (default: `1`)
        minimum number of bins possible
    max_bins: int, optional (default: `1000`)
        maximum number of bins possible
    groups: list[str], (default: `None`)
        in adata.obs; groups by all possible combinations of values, e.g. for
        3 plates and 2 time points, we would create total of 6 groups
    fill_alpha: float[0.0, 1.0], (default: `0.4`)
        alpha channel of fill color
    legend_loc: str, (default: `top_right`)
        position of the legend
    tools: str, optional (default: `"pan,reset, wheel_zoom, save"`)
        palette of interactive tools for the user
    palette: list, optional (default: `Set1[9] + Set2[8] + Set3[12]`)
         colors from bokeh.palettes, e.g. Set1[9]
    display_all: bool, optional (default: `True`)
        display the statistics for all data
    **kwargs: keyword arguments for figure
        specify e.g. `"plot_width"` to set the width of the figure.

    Returns
    --------
    None
    """

    from itertools import product
    from functools import reduce
    from bokeh.plotting import figure, show, ColumnDataSource
    from bokeh.models.widgets import CheckboxGroup
    from bokeh.models.widgets.buttons import Button
    from bokeh.models import Slider
    from bokeh.models.callbacks import CustomJS
    from bokeh.io import output_notebook
    from bokeh.layouts import layout, column, row

    from copy import copy
    from numpy import array_split, ceil
    output_notebook()

    if min_bins < 1:
        raise ValueError(f'Expected min_bins >= 1, got min_bins={min_bins}.')
    if max_bins < min_bins:
        raise ValueError(f'Expected min_bins <= max_bins, got min_bins={min_bins}, max_bins={max_bins}.')
    if not (bins >= min_bins and bins <= max_bins):
        raise ValueError(f'Expected min_bins <= bins <= max_bins, got min_bins={min_bins}, bins={bins}, max_bins={max_bins}.')

    # check the input
    for key in keys:
        if key not in adata.obs.keys() and \
           key not in adata.var.keys() and \
           key not in adata.var_names:
            raise ValueError(f'The key `{key}` does not exist in adata.obs, adata.var or adata.var_names.')

    def _create_adata_groups():
        if groups is None:
            return [('all',)], [adata]

        combs = list(product(*[set(adata.obs[g]) for g in groups]))
        adatas= [adata[reduce(lambda l, r: l & r,
                              (adata.obs[k] == v for k, v in zip(groups, vals)), True)]
                 for vals in combs] + [adata]

        if display_all:
            combs += [('all',)]
            adatas += [adata]

        return combs, adatas

    # group_v_combs contains the value combinations
    # used for grupping
    group_v_combs, adatas = _create_adata_groups()
    n_plots = len(group_v_combs)
    checkbox_group = CheckboxGroup(active=list(range(n_plots)), width=200)
    
    for key in keys:
        # create histogram
        cols, legends, callbacks = [], [], []
        plot_map = dict()
        slider = Slider(start=min_bins, end=max_bins, value=bins, step=1,
                        title='Bins')

        fig = figure(*args, tools=tools, **kwargs)

        plot_ids = []
        for j, (ad, group_vs) in enumerate(zip(adatas, group_v_combs)):

            if ad.n_obs == 0:
                continue
            
            plot_ids.append(j)
            color = palette[len(plot_ids) - 1]

            if key in ad.obs.keys():
                orig = ad.obs[key]
                hist, edges = np.histogram(orig, density=True, bins=bins)
            elif key in ad.var.keys():
                orig = ad.var[key]
                hist, edges = np.histogram(orig, density=True, bins=bins)
            else:
                orig = ad[:, key].X
                hist, edges = np.histogram(orig, density=True, bins=bins)

            # original data, used for recalculation of histogram in JS code
            orig = ColumnDataSource(data=dict(values=orig))
            # data that we update in JS code
            source = ColumnDataSource(data=dict(hist=hist, l_edges=edges[:-1], r_edges=edges[1:]))

            legend = ', '.join(': '.join(map(str, gv)) for gv in zip(groups, group_vs)) \
                    if groups is not None else 'all'
            legends.append(legend)
            # create figure
            p = fig.quad(source=source, top='hist', bottom=0,
                         left='l_edges', right='r_edges',
                         fill_color=color, legend=legend,
                         line_color="#555555", fill_alpha=fill_alpha)

            # create callback and slider
            callback = CustomJS(args=dict(source=source, orig=orig), code=_inter_hist_js_code)
            callback.args['bins'] = slider
            callbacks.append(callback)

            # add the current plot so that we can set it
            # visible/invisible in JS code
            plot_map[f'p_{j}'] = p

        # slider now updates all values
        slider.js_on_change('value', *callbacks)
        plot_map['cb'] = checkbox_group

        button = Button(label='Toggle All', button_type='primary')
        code_t='\n'.join(f'p_{p_id}.visible = false;' for i, p_id in enumerate(plot_ids))
        code_f ='\n'.join(f'p_{p_id}.visible = true;' for i, p_id in enumerate(plot_ids))
        button.callback = CustomJS(
            args=plot_map,
            code=f'''if (cb.active.length == {len(plot_map) - 1}) {{
                console.log(cb.active);
                cb.active = Array();
                {code_t};
            }} else {{
                console.log(cb.active);
                cb.active = Array.from(Array({len(plot_map) - 1}).keys());
                {code_f};
            }}'''
        )

        checkbox_group.callback = CustomJS(
            args=plot_map,
            code='\n'.join(f'p_{p_id}.visible = cb.active.includes({i});' for i, p_id in enumerate(plot_ids))
        )
        checkbox_group.labels = legends

        fig.legend.location = legend_loc
        fig.xaxis.axis_label = key
        fig.yaxis.axis_label = 'normalized frequency'
        fig.plot_width = kwargs.get('plot_width', 400)
        fig.plot_height = kwargs.get('plot_height', 400)

        cols.append(column(slider, button, row(fig, checkbox_group)))


    # transform list of pairs of figures and sliders into list of lists, where
    # each sublist has length <= 2
    # note that bokeh does not like np.arrays
        grid = list(map(list, array_split(cols, ceil(len(cols) / 2))))

        show(layout(children=grid, sizing_mode='fixed', ncols=2))


def plot_cell_indices(adata, key='group', basis='diffmap', components=[1, 2],
                      legend_loc='top_right', tools='pan, reset, wheel_zoom, save'):
    """
    Plot cell indices. Useful when trying to set adata.uns['iroot'].

    Params
    --------
    adata: AnnData Object
        annotated data object
    key: str, optional (default: `"group"`)
        key in adata.obs to color
    basis: str, optional (default: `"diffmap"`)
        basis to use
    components: list[int], optional (default: `[1, 2]`)
        which components of the basis to use
    legend_loc: str, optional (default `"top_right"`)
        location of the legend
    tools: str, optional (default: `"pan, reset, wheel_zoom, save"`)
        tools for the plot

    Returns
    --------
    None
    """
    from bokeh.layouts import column
    from bokeh.models import ColumnDataSource
    from bokeh.plotting import figure, show
    from bokeh.palettes import viridis
    from bokeh.models.widgets.buttons import Button
    from bokeh.models.callbacks import CustomJS
    from bokeh.models import LabelSet, CategoricalColorMapper

    if key not in adata.obs:
        raise ValueError(f'{key} not found in adata.obs')

    if f'X_{basis}' not in adata.obsm_keys():
        raise ValueError(f'basis `X_{basis}` not found in adata.obsm')

    if not isinstance(components, type(np.array)):
        components = np.array(components)

    df = pd.DataFrame(adata.obsm[f'X_{basis}'][:, components - (0 if basis == 'diffmap' else 1)], columns=['x', 'y'])
    df[key] = list(adata.obs[key])
    df['index'] = range(len(df))

    palette = adata.uns.get(f'{key}_colors', viridis(len(df[key].unique())))

    p = figure(title=f'{key}', tools=tools)
    key_col = adata.obs[key].astype('category') if adata.obs[key].dtype.name != 'category' else  adata.obs[key]
    for c, color in zip(key_col.cat.categories, palette):
        data = ColumnDataSource(df[df[key] == c])
        p.scatter(x='x', y='y', size=10, color=color, legend=str(c), source=data)

    p.legend.location = legend_loc
    p.xaxis.axis_label = f'{basis}_{components[0]}'
    p.yaxis.axis_label = f'{basis}_{components[1]}'

    source = ColumnDataSource(df)
    labels = LabelSet(x='x', y='y', text='index',
                      x_offset=4, y_offset=4,
                      level='glyph',
                      source=source, render_mode='canvas')
    labels.visible = False
    p.add_layout(labels)

    button = Button(label='Toggle Indices', button_type='primary')
    button.callback = CustomJS(args=dict(l=labels), code='l.visible = !l.visible;')

    show(column(button, p))


def plot_pcs(adata, pcs=[1, 2], groups=['n_counts', 'n_genes']):
    """
    Scatter plot of projected data and targets in groups.
    Params
    --------
    adata: AnnData Object
        Annotated data object
    pcs: list, optional (default: `[1, 2]`)
        projections to use
    groups: list, optional (default: `["n_counts"]`)
        keys in adata.obs_keys() or adata.var_name where targets are stored
    
    Returns
    --------
    None
    """

    if not isinstance(pcs, type(np.array)):
        pcs = np.array(pcs)

    ys = [adata.obs[g] if g in adata.obs_keys() else adata[:, g].X if g in adata.var_names else None
          for g in groups]
    
    ok = tuple(map(lambda y: y is not None, ys))
    if not all(ok):
        raise ValueError(f'Unknown groups: {np.array(groups)[~np.array(ok)]}.')
        
    proj = adata.obsm['X_pca'][:, pcs - 1]
    
    for group, y in zip(groups, ys):
        for i, pc in enumerate(pcs):
            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.set_title(f'PC_{pc}')
            ax.set_ylabel(group)

            ax.scatter(proj[:, i], y)


def plot_r2_scores(adata, components=[1, 2], groups=['n_counts', 'n_genes'],
                   basis='pca', take=None):
    """
    Fit linear models and plot R\u00B2 scores
    using projected data and targets in groups.
    Params
    --------
    adata: AnnData Object
        annotated data object
    components: list, optional (default: `[1, 2]`)
        components to use
    groups: list, optional (default: `["n_counts", "n_genes"]`)
        keys in adata.obs_keys() or adata.var_name where targets are stored
    basis: str, optional (default: `"pca"`)
        basis stored in adata.obsm
    take: int, optional (default: `None`)
        number of highest R\u00B2 scores to plot
    
    Returns
    --------
    None
    """
    if 'X_' + basis not in adata.obsm.keys():
        raise ValueError(f'You have not computed \'{basis}\' basis.')

    from sklearn.linear_model import LinearRegression
    from collections import defaultdict

    import matplotlib.cm as cm
    import matplotlib.patches as mpatches

    if not isinstance(components, type(np.array)):
        components = np.array(components)

    if isinstance(groups, dict):
        # we call them genes even though they may be something else
        cmap = {group:cm.Set1(i) for i, group in enumerate(groups.keys())}
        gene_to_group = {gene:group for group, genes in groups.items() for gene in genes}
        groups = list(gene_to_group.keys())
    else:
        cmap = defaultdict(lambda: 'black')
        gene_to_group = defaultdict(lambda: None)

    if groups is None:
        groups = np.concatenate([adata.var_names, adata.obs_keys()])

    reg = LinearRegression()
    
    proj = adata.obsm['X_' + basis][:, components - 1]
    proj = np.expand_dims(proj, axis=2)  # linreg. requires this
    
    score_groupss = (sorted(((reg.fit(x, y).score(x, y), g)
        for x, y, g in ((proj[:, i], adata[:, g].X if g in adata.var_names else adata.obs[g], g)  # this just sets variable names
            for g in groups)), reverse=True, key=lambda r_g: r_g[0])[:take]
                for i in range(components.shape[0]))
    
    len_g = len(groups)
    x_ticks = np.arange(0, len_g, max(1, min(10, len_g // 10)))
    
    for component, sgs, in zip(components, score_groupss):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        # TODO: better naming
        for ix, (score, group) in enumerate(sgs):
            ax.annotate(f'{group}', xy=(ix, score), xycoords='data',
                        color=cmap[gene_to_group[group]], rotation=90, ha='left', va='bottom')
            
        ax.set_title(f'{basis}_{component}')
        ax.xaxis.set_ticks(x_ticks)
        
        plt.xlabel('rank')
        plt.ylabel('R\u00B2 score')

        if not isinstance(cmap, defaultdict):
            plt.legend(handles=[mpatches.Patch(color=color, label=f'{group}') for group, color in cmap.items()])
        
        plt.xlim(-1, len_g)
        plt.ylim(0, 1)
        plt.grid(visible=True)
        
        plt.show()


def simple_de_matching(adata, markers, n_genes=100):
    """
    Print markers that that match genes found in
    differential expression test for each cluster.

    Params
    --------
    adata: AnnData Object
        Annotated data object
    markers: dict
        dictionary of known marker genes, e.g. for a specific cell type, each with a key
    n_genes: int, optional (default: `10`)
        number of genes to consider
    
    Returns
    --------
    de_genes : pd.DataFrame
        Contains the differentially expressed genes
    """
    gene_groups = adata.uns['rank_genes_groups']
    de_genes = pd.DataFrame(data=gene_groups['names']).head(n_genes)
    #print(de_genes.head(10))

    matches = check_markers(de_genes, markers)
    for key, value in matches.items():
        print(f'-- cluster {key} --')
        print(value)

    return de_genes


def create_cellxgene_browser(adata, token, jupyter_url='http://localhost:8888', dst_path='',
                             username='user', password='passwd', browser_name='analysis',
                             adata_fname='cellxgene.h5ad', cfg_fname='cellxgene.config',
                             ret=False):
    """
    Create an .h5ad and .config files for the cellxgene browser.

    Params
    --------
    adata: AnnData object
        annData object to be visualized.
    token: str
        Token for the Jupyter server.
    jupyter_url: str (default: `"http://localhost:8888"`)
        Url pointing to the Jupyter server.
    dst_path: str (default: `""`)
        Path to the destination directory. `""` corresponds to the root on the Jupyter server.
    username: str (default: `"user"`)
        Username for the cellxgene browser.
    password: str (default: `"passwd"`)
        Password for the cellxgene browser.
    browser_name: str (default: `"analysis"`)
        Name of the cellxgene browser.
    adata_fname: str (default: `"cellxgene.h5ad"`)
        Name of the .h5ad file.
    cfg_name: str (default: `"cellxgene.config"`)
        Name of the .config file.
    ret: bool (default: `False`)
        Whether to return output statuses for each file transfer.

    --------
    Returns: if `res` is True, then Tuple(requests.models.Response, requests.models.Response)
             otherwise None
    Responses corresponding to the creation of the .h5ad and .config files respectively.
    """

    from tempfile import NamedTemporaryFile
    from itertools import accumulate
    from time import sleep
    import os
    import warnings
    import requests
    import urllib
    import base64
    import json

    def create_folder(path, warn=False):
        dst_url= os.path.join(jupyter_url, 'api/contents/', urllib.parse.quote(path))
        resp = json.loads(session.get(dst_url, data=dict(path=path, type='directory'), params={'token': token}).text)
        if resp.get('type') is None:
            dir_data = json.dumps({
                'name': os.path.basename(path),
                'path': path,
                'format': 'json',
                'type': 'directory',
                'mimetype': None
            })
            resp = session.put(dst_url, data=dir_data, params={'token': token})
            if resp.status_code != 201:  # resource succesfully created
                raise RuntimeError(f"Creating directory `{path}` returned status code: {resp.status_code}. Message: `{json.loads(resp.text).get('message', None)}`.")
            return resp.cookies
        elif resp['type'] != 'directory':
            raise OSError(os.errno.EEXIST, f"`{path}` already exists. Expected it to be of type `directory`, got type `{resp['type']}`.")
        elif warn:
            warnings.warn(f'Directory `{path}` already exists, will overwrite files if necessary.')
        return {}

    cookies = {}
    session = requests.session()

    paths = list(accumulate([p for p in dst_path.split(os.path.sep) if p != ''], lambda p, d: os.path.join(p, d)))
    for path, warn in zip(paths, [False] * (len(paths) - 1) + [True]):
        cookies = create_folder(path, warn)

    cfg_data = json.dumps({
        'content': f'BASIC_USER={username}\nBASIC_PW={password}\nSAMPLENAME={browser_name}\n',
        'name': cfg_fname,
        'path': dst_path,
        'format': 'text',
        'type': 'file',
        'mimetype': 'text/plain'
    })

    tmp_f = NamedTemporaryFile('w', suffix='.h5ad')
    sc.write(tmp_f.name, adata)

    # just in case
    tmp_f.flush()
    tmp_f.seek(0)

    with open(tmp_f.name, 'rb') as f:
        ad_data = json.dumps({
            'content': base64.encodebytes(f.read()).decode('ascii'),
            'name': adata_fname,
            'path': dst_path,
            'format': 'base64',
            'type': 'file'
        })


    res = []
    dst_url= os.path.join(jupyter_url, 'api/contents/', urllib.parse.quote(dst_path))
    
    for fname, data in zip([adata_fname, cfg_fname], [ad_data, cfg_data]):
        res.append(session.put(os.path.join(dst_url, urllib.parse.quote(fname)),
                               data=data, params={'token': token}, cookies=cookies))
        sleep(20)

    return tuple(res) if ret else None


def plot_gene(adata, ax, x, y, type='gene', x_test=None, x_mean=None, x_cov=None, x_grad=None,
                 scatter_kwgs=None):
    """Utility function to plot gene expression, velocity and dpt
       TODO: this is merely a placeholder.

    Parameters
    --------
    adata: AnnData Object
        Annotated Data Frame
    ax: plt.axes object
        Axes used for plotting
    x: np.array
        Temporal ordering
    y: np.array
        Expression values
    type: str, optional (default: `"s"`)
        What is being plotted. Gene expression (gene) or velocity (velocity)
    x_test: np.array, optional (default: `None`)
        Grid of values for testing
    x_mean: np.array, optional (default: `None`)
        Smoothed expression values
    x_cov: np.array, optional (default: `None`)
        Covariance matrix for smoothed expression
    x_grad: np.array, optional (default: `None`)
        Derivative of gene expression
    gene_name: str, optional (default: `" "`)
        Name of the gene for plotting
    scatter_kwgs: None or dict, optional (default: `None`)
        keyword arguments for scv.pl.scatter

    Returns
    --------
    Nothing, just plots.
    """

    # basic scatter plot
    if scatter_kwgs is not None:
        ax = scv.pl.scatter(adata, x=x, y=y, ax=ax, show=False, **scatter_kwgs)
    else:
        ax = scv.pl.scatter(adata, x=x, y=y, ax=ax, show=False)
    ax.set_title(" ")

    if x_test is not None:
        # add smoothed expression valued
        if x_mean is not None:
            ax.plot(x_test, x_mean, '-', color='orange', lw=3,
                    label='Smoothed {} expression values'.format(type))
            # add covariance
            if x_cov is not None:
                ax.fill_between(x_test.flatten(), x_mean - np.sqrt(np.diag(x_cov)),
                                x_mean + np.sqrt(np.diag(x_cov)),
                                alpha=0.5, color='k')
        # add the derivative
        if x_grad is not None:
            ax.plot(x_test, x_grad, '--', color='orange', lw=3,
                    label='Derivative of gene expression')

    ax.set_ylabel('{} expression'.format(type), fontsize=10)
    ax.set_xticks([])
    plt.legend(fontsize=10)
