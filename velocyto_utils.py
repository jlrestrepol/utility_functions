# utilities for velocyto

"""some of these functions are just copied from the jupyter notebooks from the 
velocyto github repo, see https://github.com/velocyto-team/velocyto-notebooks/ tree/master/python"""

# import some packeges
import velocyto as vcy
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from anndata import AnnData
import pandas as pd
import scipy as scp

#%% plotting utility functions




def despline():
    ax1 = plt.gca()
    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    
def minimal_xticks(start, end):
    end_ = np.around(end, -int(np.log10(end))+1)
    xlims = np.linspace(start, end_, 5)
    xlims_tx = [""]*len(xlims)
    xlims_tx[0], xlims_tx[-1] = f"{xlims[0]:.0f}", f"{xlims[-1]:.02f}"
    plt.xticks(xlims, xlims_tx)

    
def minimal_yticks(start, end):
    end_ = np.around(end, -int(np.log10(end))+1)
    ylims = np.linspace(start, end_, 5)
    ylims_tx = [""]*len(ylims)
    ylims_tx[0], ylims_tx[-1] = f"{ylims[0]:.0f}", f"{ylims[-1]:.02f}"
    plt.yticks(ylims, ylims_tx)
    
def plot_legend(colors_dict, fontsize = 15, markerscale = 5):
        size = 5
        lp = lambda key, color: plt.plot([],color=color, ms=np.sqrt(size), mec="none",
                                label=key, ls="", marker="o")[0]
        
        handles = [lp(key, color) for key, color in colors_dict.items()]
        plt.legend(handles=handles, fontsize = fontsize, markerscale = markerscale)
        
        
def set_clusters(adata, vlm, key = 'louvain'):
    """This is a utility function which saves cluster annotations
    in a vlm object, which it takes form categorical annotations 
    in the parallel adata object
    
    Parameters
    --------
    key: str
        annotation from adata.obs to be used for cluster assignment
        
    Output
    --------
    colors_dict: dict
        stores cluser names and corresponding colors, used for legend plotting
    """
    # check the input
    if key not in adata.obs.columns:
        raise ValueError('This key does not exist in adata.obs')
    
    if key + '_colors' not in adata.uns:
        print('No corresponding colors found in adata.uns. Using whatever is in vlm')
        cluster_colors = np.unique(vlm.colorandum)
    else:
        # get the cluster colors
        cluster_colors = adata.uns[key + '_colors']
    
    # get the cluster assignment
    clusters = adata.obs[key]
    
    # get the cluster names
    #cluster_names = clusters.cat.categories
    cluster_names = np.unique(clusters.values)
    
    # construct a dict for the colors
    colors_dict = {key: cluster_colors[i] \
                   for i, key in enumerate(cluster_names)}
    
    # assign to the vlm object
    vlm.set_clusters(clusters.values, cluster_colors_dict=colors_dict)
    
    return colors_dict
    

#%% Plotting functions
    
def plot_genes_velocity(vlm, genes):
    """This function plots the distribution of spliced and unspliced counts for 
    a given gene, as well as the estimated steads state, and velocity as color on the embedding. 
    
    Parameters
    --------
    vlm: VelocytoLoom object
    genes: list
        list of genes to consider
    """
    
    # visualise

    n_genes = len(genes)
    n_row = np.int(np.ceil(n_genes/2))
    n_col = 6
    plt.figure(None, (17, 2.8* n_row), dpi = 80)
    gs = plt.GridSpec(n_row, n_col)

    for i, gn in enumerate(genes):
        ax = plt.subplot(gs[i*3])
        try:
            ix = np.where(vlm.ra["Gene"] == gn)[0][0]
        except:
            continue
        # make a scatter plot of spliced and unspliced counts
        vcy.scatter_viz(vlm.Sx_sz[ix, :], vlm.Ux_sz[ix, :], 
                        c = vlm.colorandum, s = 5, alpha = 0.4, 
                       rasterized = True)    
        plt.title(gn)
        plt.xlabel('Spliced')
        plt.ylabel('Unspliced')

        # add the trend showing the estimated steadt state
        xnew = np.linspace(0, vlm.Sx[ix, :].max())
        plt.plot(xnew, vlm.gammas[ix] *xnew + vlm.q[ix], c = 'k')

        # change the axis limits
        plt.ylim(0, np.max(vlm.Ux_sz[ix, :])* 1.02)
        plt.xlim(0, np.max(vlm.Sx_sz[ix, :])*1.02)

        # have fewer ticks on the yaxis
        minimal_yticks(0, np.max(vlm.Ux_sz[ix, :])* 1.02)
        minimal_xticks(0, np.max(vlm.Sx_sz[ix, :])* 1.02)
        # get rid of the top and right axis
        despline()

        # plot velocoties
        vlm.plot_velocity_as_color(gene_name = gn, gs= gs[i*3 + 1], 
                                  s = 3, rasterized= True)
        vlm.plot_expression_as_color(gene_name=gn, gs = gs[i*3+2], s = 3, rasterized = True)

        plt.tight_layout()
        
def plot_arrows(vlm, colors_dict = None, fontsize = 10, quiver_scale = 10, axis_on = 'off', 
                filename = None, xlabel = '', ylabel = '', dpi = 100, 
                figsize = (5, 5), markerscale = 5):
    """
    Plot the velocity arrows onto the embedding saved in vlm.ts
    
    Parameters 
    --------
    vlm: Velocyto Loom object
    colors_dict: dict
        Contains colors for the cluster names. When present, a legend is added.
    fontsize: str
        legend fontsize
    """
    
    plt.figure(None, figsize = figsize , dpi = dpi)

    plt.scatter(vlm.embedding[:, 0], vlm.embedding[:, 1],
                c="0.8", alpha=0.2, s=10, edgecolor="")

    ix_choice = np.random.choice(vlm.embedding.shape[0], size=int(vlm.embedding.shape[0]/1.), replace=False)
    plt.scatter(vlm.embedding[ix_choice, 0], vlm.embedding[ix_choice, 1],
                color = vlm.colorandum[ix_choice],
                alpha=0.2, s=20, edgecolor=(0,0,0,1), lw=0.3, rasterized=True)

    quiver_kwargs=dict(headaxislength=10, headlength=11, headwidth=12,linewidths=0.5, width=0.00045,edgecolors="k", color=vlm.colorandum[ix_choice], alpha=1)
    plt.quiver(vlm.embedding[ix_choice, 0], vlm.embedding[ix_choice, 1],
               vlm.delta_embedding[ix_choice, 0], vlm.delta_embedding[ix_choice, 1],
               scale=quiver_scale, **quiver_kwargs)
    
    
    # add a legend
    if colors_dict is not None:
        plot_legend(colors_dict, fontsize = fontsize, markerscale=markerscale)
        

    plt.axis(axis_on)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot()
    if filename is not None:
        plt.savefig(filename, dpi = dpi , format = 'svg', bbox_inches = 'tight', pad_inches = 0)
    

 
    
    
def plot_grid_arrows(vlm, colors_dict = None, fontsize = 10, min_mass = 5, quiver_scale = 0.7):
    
    # plot smoothed velocities on the grid
    plt.figure(None, (10, 10), dpi = 200)
    
    # keyword arguments for the scatter plot
    scatter_kwags = {'alpha': 0.35, "lw": 0.35,
                     "edgecolor":"0.4", 
                    "s": 38, "rasterized": True}
    
    # the rasrized keyword just means that the plot will
    # be compressed somehow to reduce storage
    vlm.plot_grid_arrows(quiver_scale=quiver_scale, 
                         scatter_kwargs_dict=scatter_kwags,
                        min_mass = min_mass, angles = 'xy', 
                         scale_units = 'xy', 
                         headaxislength = 2.75, 
                        headlength = 5, headwidth = 4.8,
                        minlength = 1.5, 
                        scale_type = "absolute")
    # add a legend
    if colors_dict is not None:
        plot_legend(colors_dict, fontsize = fontsize)

    plt.axis("off")
    plt.plot()
    
    
def plot_arrows_zoom(vlm, axis_ranges = None, colors_dict = None, plot_title = '', fontsize = 15, figsize = (6, 6)):
    """
    Plots a region of the embeddign with a selected number of cells and their velocities.
    
    Parameters
    --------
    vlm: VelocytoLoom object
    axis_ranges: list
        shoul be [xmin, xmax, ymin, xmax]
    colors_dict: dict
        used for the legend, should contain colors for the cells displayed
    """
    plt.figure(None, figsize)
    
    # this is not really a gaussian kernel but more a
    # gaussian distribution. we use it for local density estimation
    def gaussian_kernel(X, mu = 0, sigma = 1):
        return np.exp(-(X - mu)**2 / (2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)
    
    steps = 45, 45
    grs = []
    
    
    for dim_i in range(vlm.embedding.shape[1]):
        
        # get the range of the embeddig distr in every dim (2)
        m, M = np.min(vlm.embedding[:, dim_i]), \
        np.max(vlm.embedding[:, dim_i])
        
        # widen the range slightly
        m = m - 0.025 * np.abs(M - m)
        M = M + 0.025 * np.abs(M - m)
        
        # create a grid for each dimension, spanning the range
        gr = np.linspace(m, M, steps[dim_i])
        grs.append(gr)
        
    # build a mesh grid from these lists
    # the * operator unpacks the lists
    meshes_tuple = np.meshgrid(*grs)
    
    # put the gridpoint coordinates in vectors and stack them together
    gridpoint_coordinates = np.vstack([i.flat \
                                       for i in meshes_tuple]).T
    # add some normal noise to the gridpoint coordinates
    gridpoint_coordinates = gridpoint_coordinates + \
    norm.rvs(loc = 0, scale = 0, size = gridpoint_coordinates.shape)
     
    nn = NearestNeighbors()
    # train the nearest neighboors classifier
    nn.fit(vlm.embedding)
    # for each grid point, we find the 20 closest cells aroundit
    dist, ixs = nn.kneighbors(gridpoint_coordinates, 20)
    ix_choice = ixs[:, 0].flat[:]
    ix_choice = np.unique(ix_choice)
    
    # refine the choice of cells based on the local density
    nn = NearestNeighbors()
    nn.fit(vlm.embedding)
    # for our previously chosen cells, find nearest neighbors
    # in the full data set
    dist, ixs = nn.kneighbors(vlm.embedding[ix_choice], 20)
    # estimate the density around each of our chosen cells
    density_extimate = gaussian_kernel(dist, mu=0, sigma=0.5).sum(1)
    # find dense regions
    bool_density = density_extimate > np.percentile(density_extimate, 25)
    ix_choice = ix_choice[bool_density]
    
    # plot all points blurry
    plt.scatter(vlm.embedding[:, 0], vlm.embedding[:, 1],
                c=vlm.colorandum, alpha=0.2, s=120, edgecolor="")
    
    # plot the selected points
    plt.scatter(vlm.embedding[ix_choice, 0], vlm.embedding[ix_choice, 1],
                c=vlm.colorandum[ix_choice], alpha=1, s=120, edgecolor="k")
    
    # plot arrows for these cells
    quiver_kwargs=dict(scale=6.8, headaxislength=9, headlength=15, headwidth=14,linewidths=0.4, edgecolors="k", color="k", alpha=1)
    plt.quiver(vlm.embedding[ix_choice, 0], vlm.embedding[ix_choice, 1],
               vlm.delta_embedding[ix_choice, 0], vlm.delta_embedding[ix_choice, 1],
               **quiver_kwargs)
    
    # focus on one region
    if axis_ranges is not None:
        plt.xlim(axis_ranges[:2])
        plt.ylim(axis_ranges[2:])
    
    despline()
    plt.title(plot_title)
    
    if colors_dict is not None:
        plot_legend(colors_dict, fontsize = fontsize)
        
    plt.plot()


def plot_selected_transitions(vlm, ix1 = 1, ix2 = 2, axis_range = None):
    """Plots transition probabilities for neighboring cells, based on two user-defined cells
    
    Parameters
    --------
    vlm: VelocytoLoom Object
    ix1, ix2: int
        indices of the two cells
    axis_range: list
        should be [xmin, xmax, ymin, ymax]
    """

    plt.figure(None,(10,3), dpi=130)

    def plot_trans_prob(ix, axis_range):

        # chose the neighboring points
        neigh_bool = (vlm.embedding_knn[ix, :].A[0] > 0 )

        # get the transition probabilities to these neighboring cells
        colorandum = vlm.transition_prob[ix, :][neigh_bool]

        # transform the transition probabilities for plotting
        colorandum -= 1e-6 
        colorandum = colorandum / 7e-4
        colorandum = np.clip(colorandum, 0, 1)

        # plot the neighboring cells and color them according to the 
        # transition probability
        p = np.argsort(colorandum)
        ax = plt.scatter(vlm.embedding[neigh_bool, 0][p],
                   vlm.embedding[neigh_bool, 1][p],
                   c = colorandum[p], 
                   cmap = plt.cm.viridis, alpha = 0.4, 
                   lw = 0.7, s = 50, edgecolor = '0.5')
        plt.scatter(vlm.embedding[ix, 0], vlm.embedding[ix, 1],
                    c="r", alpha=1, s=150, lw=3, edgecolor="0.8", marker="D")
        plt.title('Trans. prob for cell ' + str(ix))
        if axis_range is not None:
            plt.xlim(axis_range[:2])
            plt.ylim(axis_range[2:])

        return ax

    # plot transition probebilities for both cells
    plt.subplot(131)
    _= plot_trans_prob(ix1, axis_range)
    despline()

    plt.subplot(132)
    ax = plot_trans_prob(ix2, axis_range)
    plt.colorbar(ax)
    despline()

    # add a colorbar for both of these
    plt.subplot(133)
    plt.scatter(vlm.embedding[:, 0], vlm.embedding[:, 1], 
               c = vlm.colorandum)


    plt.scatter(vlm.embedding[ix1, 0], vlm.embedding[ix1, 1], 
               c = 'r', alpha = 1, s = 150, lw = 3, 
                edgecolor = '0.8', marker = 'D')
    plt.scatter(vlm.embedding[ix2, 0], vlm.embedding[ix2, 1], 
               c = 'r', alpha = 1, s = 150, lw = 3, 
                edgecolor = '0.8', marker = 'D')
    plt.axis("off")
    
    plt.title('Complete Embedding')


    plt.plot()
    
    
#%% General utility functions
    
def sample_down(X_em, n_points):
    """
    Downsampling of the number of cells in a given embedding to avoid
    density driven effects. The number of dimensions in the embedding 
    determines the scpace searched over. In higher dimenisons, the number
    of grid points per dimension shoud be reduced.
    
    Parameters
    -------
    X_em: np.array
        Embedding coordinates
    n_points: int, tupel, list
        number of grid points per dimension in the embedding. 
        
    Output
    --------
    ixs: list
        indices of the remaining cells
    diag_step_dist: float
        indicates the diagonal distance in the high dimensional mesh
    """
    
    # set up the grids for each dimension
    grs = []
    dists = []

    # for each dimension, compute a regular grid covering the whole data range
    for dim_i in range(X_em.shape[1]):
        m, M = np.min(X_em[:, dim_i]), np.max(X_em[:, dim_i])
        m = m - 0.025 * np.abs(M - m)
        M = M + 0.025 * np.abs(M - m)
        gr = np.linspace(m, M, n_points[dim_i])
        grs.append(gr)
        dists.append((M - m)/(n_points[dim_i]- 1))

    # compute a meshgrid. meshes tuple contains two matrices
    meshes = np.meshgrid(*grs)

    # reshape the matrix into one long vector
    gridpoints_coordinates = np.vstack([i.flat for i in meshes]).T


    # initialise a nearest neighboors object
    nn = NearestNeighbors(n_neighbors=1, n_jobs = 4)

    # fit a nearest neighboor classifier in the embedding
    nn.fit(X_em)

    # for each point in the grid, find the nearest neighboor, and the distance to that neighboor
    dist, ixs = nn.kneighbors(gridpoints_coordinates, 1)

    # compute the diagonal distance
    diag_step_dist = np.linalg.norm(dists)
    min_dist = diag_step_dist / 2

    # only keep the indices of cells which were close to the reference points
    ixs = ixs[dist < min_dist]

    # only keep grid points and distances corresponding to these cases
    gridpoints_coordinates = gridpoints_coordinates[dist.flat[:]<min_dist,:]
    dist = dist[dist < min_dist]

    # for each grid point, only keep one reference cell
    ixs = np.unique(ixs)

    return ixs, diag_step_dist

def vlm_to_adata(vlm, trans_mats = None, cells_ixs = None, em_key = None):
    """ Conversion function from the velocyto world to the scanpy world
    
    Parameters
    --------
    vlm: VelocytoLoom Object
    trans_mats: None or dict
        A dict of all relevant transition matrices
    cell_ixs: list of int
        These are the indices of the subsampled cells
    
    Output
    adata: AnnData object
    """
    
    # create the anndata object
    adata = AnnData(
        vlm.Sx_sz.T, vlm.ca, vlm.ra,
        layers=dict(
            unspliced=vlm.U.T,
            spliced = vlm.S.T, 
            velocity = vlm.velocity.T),
        uns = dict(velocity_graph = vlm.corrcoef, louvain_colors = list(np.unique(vlm.colorandum)))
    )
        
    # add uns annotations
    if trans_mats is not None:
        for key, value in trans_mats.items():
            adata.uns[key] = trans_mats[key]
    if cells_ixs is not None:
        adata.uns['cell_ixs'] = cells_ixs
        
    # rename clusters to louvain
    try:
        ix = np.where(adata.obs.columns == 'Clusters')[0][0]
        obs_names = list(adata.obs.columns)
        obs_names[ix] = 'louvain'
        adata.obs.columns = obs_names
        
        # make louvain a categorical field
        adata.obs['louvain'] = pd.Categorical(adata.obs['louvain'])
    except:
        print('Could not find a filed \'Clusters\' in vlm.ca.')
            
    # save the pca embedding
    adata.obsm['X_pca'] = vlm.pcs[:, range(50)]
    
    # transfer the embedding
    if em_key is not None:
        adata.obsm['X_' + em_key] = vlm.ts
        adata.obsm['velocity_' + em_key] = vlm.delta_embedding
    
    # make things sparse
    adata.X = scp.sparse.csr_matrix(adata.X)
    adata.uns['velocity_graph'] =scp.sparse.csr_matrix(adata.uns['velocity_graph'])
    
    # make the layers sparse
    adata.layers['unspliced'] = scp.sparse.csr_matrix(adata.layers['unspliced'])
    adata.layers['spliced'] = scp.sparse.csr_matrix(adata.layers['unspliced'])
    adata.layers['velocity'] = scp.sparse.csr_matrix(adata.layers['unspliced'])
    
    return adata



    
    
    
    
    
