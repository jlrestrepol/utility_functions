def timeseries_smooth(adata, genes= 'none', gene_symbols= 'none', 
               key = 'louvain' , groups = 'all', style = '-b', 
              n_restarts_optimizer = 10, likelihood_landscape = False, normalize_y=False,
              noise_level = 0.5, noise_level_bounds = (1e-2, 1e+1), 
              length_scale = 1, length_scale_bounds = (1e-2, 1e+1), 
              save = 'none', title = 'long'):
    
    """
    Plot a timeseries of some genes in pseudotime
    
    Keyword arguments:
    adata -- anndata object
    genes -- list of genes. If 'none', the first 5 genes are plotted
    gene_symbols -- variable annotation. If 'none', the index is used
    key -- observation annotation. 
    groups -- basically branches, chosen from the annotations in key
    style -- line plotting style
    """
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
    
    # select one branch
    if groups != 'all':
        adata_selected = adata[np.in1d(adata.obs[key], groups)]
    else:
        adata_selected = adata
        
    # select genes
    if genes == 'none':
        
        # no genes specified, we just use the first 5
        genes = adata_selected.var_names.values[0:5]
        m = {'Mapped': genes, 'Original': genes}
        
    elif gene_symbols != 'none':
        
        # a gene annotation is used, we map the gene names
        mapping_table = pd.DataFrame(adata_selected.var[gene_symbols])
        name_mapping = mapping_table.set_index(gene_symbols)
        name_mapping['Ensembl'] = mapping_table.index
        genes_mapped = name_mapping.loc[genes, :]
        
        
        # save in dict
        m = {'Mapped': genes_mapped['Ensembl'], 'Original': genes}
    else:
        m = {'Mapped': genes, 'Original': genes}
        
    # construct a look up table
    gene_table = pd.DataFrame(data = m)
    
    # extract the pseudotime
    time = adata_selected.obs['dpt_pseudotime']
    
    
    # construct a data frame which has time as index
    exp_data = pd.DataFrame(data = adata_selected[:, gene_table['Mapped']].X, 
                            index = time, columns=[gene_table['Original'].values])
    
    # sort according to pseudotime
    exp_data.sort_index(inplace = True)
    ()
    
    # remove the last entry
    (m, n) = exp_data.shape
    exp_data = exp_data.iloc[:m-1, :]
    
    # loop counter
    i = 0

    # loop over all genes we wish to plot
    for index, row in gene_table.iterrows():   
        
        # select data
        data_selected = exp_data.loc[:, row['Original']].reset_index()
        
        # create the labels
        X = np.atleast_2d(data_selected['dpt_pseudotime'].values)
       
        # create the targets
        y = data_selected[row['Original']].values.ravel()
        
        # Mesh the input space for evaluations of the prediction and
        # its MSE
        x = np.atleast_2d(np.linspace(0, 1, 1000)).T
        
        # Initiate a Gaussian process modell. We use a sum of two kernels here, this allows 
        # us to estimate the noice level via optimisation of the marginal likelihood as well
        kernel = 1.0 * RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds) \
        + WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_level_bounds)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, 
                                      n_restarts_optimizer=n_restarts_optimizer, 
                                    normalize_y=normalize_y).fit(X, y)
        
        # obtain a prediction from this model. Also return the covariance matrix, so we can calculate
        # confidence intervals
        y_mean, y_cov = gp.predict(x, return_cov=True)
        
        # plot current genes
        plt.figure(num=i, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(x, y_mean, 'k', lw=3, zorder=9, label = 'Prediction')
        plt.fill_between(x.ravel(), y_mean - np.sqrt(np.diag(y_cov)),
                 y_mean + np.sqrt(np.diag(y_cov)),
                 alpha=0.5, color='k')
        plt.scatter(X, y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0), label= 'Observation')
        if title == 'long':
            plt.title("Gene: %s\nInitial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"
                      % (row['Original'], kernel, gp.kernel_,
                         gp.log_marginal_likelihood(gp.kernel_.theta)))
        else:
             plt.title("Gene: %s"
                       % (row['Original']))   
        plt.xlabel('$t_{pseudo}$')
        plt.ylabel('Expression')
        plt.legend(loc='upper left')
        if save != 'none':
            plt.savefig(save + row['Original'] + '_dynamics.pdf')
        
        
        if likelihood_landscape == True:
        
        # Plot LML landscape
            i += 1
            plt.figure(num=i, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
            theta0 = np.logspace(-2, 3, 49) # length scale
            theta1 = np.logspace(-1.5, 0, 50) # Noise level
            Theta0, Theta1 = np.meshgrid(theta0, theta1)
            LML = [[gp.log_marginal_likelihood(np.log([0.36, Theta0[i, j], Theta1[i, j]]))
                    for i in range(Theta0.shape[0])] for j in range(Theta0.shape[1])]
            LML = np.array(LML).T

            vmin, vmax = (-LML).min(), (-LML).max()
            #vmax = 50
            level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 50), decimals=1)
            plt.contour(Theta0, Theta1, -LML,
                    levels=level, norm=colors.LogNorm(vmin=vmin, vmax=vmax))
            plt.colorbar()
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Length-scale")
            plt.ylabel("Noise-level")
            plt.title("Log-marginal-likelihood")
            #plt.tight_layout()
            plt.show()
            
        # increase loop counter
        i += 1
        