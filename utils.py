import pandas as pd
import numpy as np
import scanpy.api as sc
import matplotlib.pyplot as plt
import re
from pybiomart import Server 

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
    
    
def print_numbers(adata, groupby = 'batch'):
    """
    Utility function to print cell numbers per batch
    
    Useful when filtering to check at intermediate steps how many cells and genes are left
    
    Parameters:
    --------
    adata : AnnData object
        Annotated data matrix
    groupby : `str`, optional (defalut: `"batch"`)
        Key to categorical annotation in adata.obs
    """
    
    # get the levels
    levels = adata.obs[groupby].cat.categories
    
    # print number of cell per batch
    for level in levels:
        n_cells = adata[adata.obs[groupby] == level].n_obs
        print('{} cells in batch {}'.format(n_cells, level))
    
    # number of genes
    print('Total: {} cells, {} genes'.\
          format(adata.n_obs, adata.n_vars))
    
    
def corr_ann(adata, obs_keys=['n_counts'], basis='pca', components=[0, 1]):
    """
    Utility function to correlate continious annoations against embedding
    
    Can be used to see how large the linear influence of a measure like the counts depth is on a 
    given component of any embedding, like PCA
    
    Parameters
    --------
    adata : AnnData object
        Annotatied data matrix
    obs_keys : `list`, optional (default: `["n_counts"]`)
        Key for the continious annotaiton to use
    basis : `str`, optional (default: `"pca"`)
        Key to the basis stored in adata.obsm
    component : `list`, optional (default: `[1, 2]`)
        Component of the embedding to use
        
    
    Returns
    --------
    Nothing, but prints the correlation
    """
    
    # check input
    if 'X_' + basis not in adata.obsm.keys():
        raise ValueError('You have not computed this basis yet')
    for obs_key in obs_keys:
        if obs_key not in adata.obs.keys():
            raise ValueError('The key \'{}\' does not exist in adata.obs'.format(obs_key))
        
    
    # get the continious annotation
    for obs_key in obs_keys:
        ann = adata.obs[obs_key]

        for component in components:
            # get the embedding coordinate
            X_em = adata.obsm['X_' + basis]
            X_em = X_em[:, component]
        
            # compute the correlation coefficient
            corr = np.corrcoef(X_em, ann)[0, 1]
            print('Correlation between \'{}\' and component \'{}\' of basis \'{}\' is {:.2f}.'.format(obs_key, 
                component + 1, basis, corr))


# quantify the batch effect quickly using the silhouette coefficient
def quant_batch(adata, key = 'batch', basis = 'pca'):
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
        
    # get the embedding coordinate
    X_em = adata.obsm['X_' + basis]
    X_em = X_em[:, :2]
    
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
                           bins=100, 
                           tools="pan,reset, wheel_zoom, save",
                           *args, **kwargs):
    """Utility function to plot count distributions\
    
    Uses the bokey library to create interactive histograms, which can be used
    e.g. to set filtering thresholds.
    
    Params
    --------
    adata: AnnData Object
        Annotated data object
    keys: list, optional (default: `["n_counts", "n_genes"]`)
        keys in adata.obs or adata.var where the distibutions are stored
    bins: int, optional (default: `100`)
        number of bins used for plotting
    tools: str, optional (default: `"pan,reset, wheel_zoom, save"`)
        palette of interactive tools for the user
    
    Returns
    --------
    Nothing
    """
    
    # import the library
    from bokeh.plotting import figure, show
    from bokeh.io import output_notebook
    from bokeh.layouts import gridplot
    output_notebook()
    
    # check the input
    for key in keys:
        if key not in adata.obs.keys() and \
           key not in adata.var.keys() and \
           key not in adata.var_names:
            raise ValueError(f'The key `{key}` does not exist in adata.obs, adata.var or adata.var_names.')
            
    # initialise lists
    figures = []
    
    # crate the histograms and the figures
    for i, key in enumerate(keys):
        
        # check wether a new sub-list must be opened
        if i == 0:
            current_list = []
        elif i%2 == 0:
            figures.append(current_list)
            current_list = []
        
        # create histogram
        if key in adata.obs.keys():
            hist, edges = np.histogram(adata.obs[key], density=True, bins=bins)
        elif key in adata.var.keys():
            hist, edges = np.histogram(adata.var[key], density=True, bins=bins)
        else:
            hist, edges = np.histogram(adata[:, key].X, density=True, bins=bins)
            
        # create figure
        fig = figure(*args, tools=tools, title=kwargs.get('title', key))
        fig.quad(top = hist, 
               bottom = 0, 
               left = edges[:-1], 
               right = edges[1:], 
               line_color = "#555555")
        fig.xaxis.axis_label = key
        fig.yaxis.axis_label = 'normalised frequency'
        current_list.append(fig)
    figures.append(current_list)
        
    # show the plots
    show(gridplot(figures, plot_width=400, plot_height=400))


def interactive_histograms_v2(adata, keys=['n_counts', 'n_genes'],
                              bins=100, min_bins=1, max_bins=1000,
                              tools='pan,reset, wheel_zoom, save',
                              *args, **kwargs):
    """Utility function to plot count distributions\

    Uses the bokey library to create interactive histograms, which can be used
    e.g. to set filtering thresholds.

    Params
    --------
    adata: AnnData Object
        Annotated data object
    keys: list, optional (default: `["n_counts", "n_genes"]`)
        keys in adata.obs or adata.var where the distibutions are stored
    bins: int, optional (default: `100`)
        number of bins used for plotting
    min_bins: int, optional (default: `1`)
        minimum number of bins possible
    max_bins: int, optional (default: `1000`)
        maximum number of bins possible
    tools: str, optional (default: `"pan,reset, wheel_zoom, save"`)
        palette of interactive tools for the user

    Returns
    --------
    None
    """

    # import the library
    from bokeh.plotting import figure, show, ColumnDataSource
    from bokeh.models import CustomJS, Slider
    from bokeh.io import output_notebook
    from bokeh.layouts import layout, column

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

    # initialize list of slider, figure pairs
    cols = []
    # crate the histograms and the figures
    for i, key in enumerate(keys):
        # create histogram
        if key in adata.obs.keys():
            orig = adata.obs[key]
            hist, edges = np.histogram(orig, density=True, bins=bins)
        elif key in adata.var.keys():
            orig = adata.var[key]
            hist, edges = np.histogram(orig, density=True, bins=bins)
        else:
            orig = adata[:, key].X
            hist, edges = np.histogram(orig, density=True, bins=bins)

        # original data, used for recalculation of histogram in JS code
        orig = ColumnDataSource(data=dict(values=orig))
        # data that we update in JS code
        source = ColumnDataSource(data=dict(hist=hist, l_edges=edges[:-1], r_edges=edges[1:]))

        # create figure
        fig = figure(*args, tools=tools, title=kwargs.get('title', key))
        fig.quad(source=source, top='hist', bottom=0,
                 left='l_edges', right='r_edges',
                 line_color="#555555")
        fig.xaxis.axis_label = key
        fig.yaxis.axis_label = 'normalised frequency'
        fig.width = 400
        fig.height = 400

        # create callback and slider
        callback = CustomJS(args=dict(source=source, orig=orig), code=_inter_hist_js_code)
        slider = Slider(start=min_bins, end=max_bins, value=bins, step=1,
                        title='Bins', callback=callback)
        callback.args['bins'] = slider

        cols.append(column(slider, fig))

    # transform list of pairs of figures and sliders into list of lists, where
    # each sublist has length <= 2
    # note that bokeh does not like np.arrays
    grid = list(map(list, array_split(cols, ceil(len(cols) / 2))))

    show(layout(children=grid, sizing_mode='fixed', ncols=2))


def plot_pcs(adata, pcs=[1, 2], groups=['n_counts']):
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


# def corr_ann(adata, obs_keys=['n_counts'], basis='pca', components=[0, 1]):
def plot_r2_scores(adata, groups=['n_counts'], basis='pca', components=[0, 1], take=None):
    """
    Fit linear models and plot R\u00B2 scores
    using projected data and targets in groups.
    Params
    --------
    adata: AnnData Object
        Annotated data object
    pcs: list, optional (default: `[1, 2]`)
        projections to use
    groups: list, optional (default: `["n_counts"]`)
        keys in adata.obs_keys() or adata.var_name where targets are stored
    take: int, optional (default: `None`)
        number of highest R\u00B2 scores to plot
    
    Returns
    --------
    None
    """

    from sklearn.linear_model import LinearRegression

    if not isinstance(components, type(np.array)):
        components = np.array(components)
        
    if groups is None:
        groups = np.concatenate([adata.var_names, adata.obs_keys()])

    reg = LinearRegression()
    
    proj = adata.obsm['X_' + basis][:, components]
    proj = np.expand_dims(proj, axis=2)  # linreg. requires this
    
    scoress = (sorted((reg.fit(x, y).score(x, y)
                          for x, y in ((proj[:, i], adata[:, g].X if g in adata.var_names else adata.obs[g])
                                       for g in groups)), reverse=True)[:take]
                  for i in range(components.shape[0]))
    
    len_g = len(groups)
    x_ticks = np.arange(0, len_g, max(1, min(10, len_g // 10)))
    
    for pc, scores in zip(components, scoress):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        for ix, (group, score) in enumerate(zip(groups, scores)):
            ax.annotate(f'{group}', xy=(ix, score), xycoords='data',
                        rotation=90, ha='left', va='bottom')
            
        ax.set_title(f'{basis}_{pc + 1}')
        ax.xaxis.set_ticks(x_ticks)
        
        plt.xlabel('rank')
        plt.ylabel('R\u00B2 score')
        
        plt.xlim(-1, len_g)
        plt.ylim(0, 1)
        plt.grid(visible=False)
        
        plt.grid()
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
    None
    """
    gene_groups = adata.uns['rank_genes_groups']
    de_genes = pd.DataFrame(data=gene_groups['names']).head(n_genes)
    print(de_genes.head(10))

    matches = check_markers(de_genes, markers)
    for key, value in matches.items():
        print(f'-- cluster {key} --')
        print(value)
