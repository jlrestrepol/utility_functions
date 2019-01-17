import pandas as pd
import patsy
import sys
import numpy.linalg as la
import numpy as np
import scipy
import scanpy.api as sc
from scipy.sparse import issparse

"""This code is mostly copied from Brent Pedersens Github repo, see https://github.com/brentp/combat.py
"""


def adjust_nums(numerical_covariates, drop_idxs):
    # if we dropped some values, have to adjust those with a larger index.
    if numerical_covariates is None: return drop_idxs
    return [nc - sum(nc < di for di in drop_idxs) for nc in numerical_covariates]

def design_mat(mod, numerical_covariates, batch_levels):
    # require levels to make sure they are in the same order as we use in the
    # rest of the script.
    design = patsy.dmatrix("~ 0 + C(batch, levels=%s)" % str(batch_levels),
                                                  mod, return_type="dataframe")

    mod = mod.drop(["batch"], axis=1)
    numerical_covariates = list(numerical_covariates)
    sys.stderr.write("found %i batches\n" % design.shape[1])
    other_cols = [c for i, c in enumerate(mod.columns)
                  if not i in numerical_covariates]
    factor_matrix = mod[other_cols]
    design = pd.concat((design, factor_matrix), axis=1)
    if numerical_covariates is not None:
        sys.stderr.write("found %i numerical covariates...\n"
                            % len(numerical_covariates))
        for i, nC in enumerate(numerical_covariates):
            cname = mod.columns[nC]
            sys.stderr.write("\t{0}\n".format(cname))
            design[cname] = mod[mod.columns[nC]]
    sys.stderr.write("found %i categorical variables:" % len(other_cols))
    sys.stderr.write("\t" + ", ".join(other_cols) + '\n')
    return design


def combat(adata, batch, model=None, numerical_covariates=None):
    """Correct for batch effects in a dataset. Expexts normalised, filtered data which
    has not yet been variance stabilised or log transformed.

    Parameters
    ----------
    data : pandas.DataFrame
        A (n_features, n_samples) dataframe of the expression or methylation
        data to batch correct
    batch : pandas.Series
        A column corresponding to the batches in the data, with index same as
        the columns that appear in ``data``
    model : patsy.design_info.DesignMatrix, optional
        A model matrix describing metadata on the samples which could be
        causing batch effects. If not provided, then will attempt to coarsely
        correct just from the information provided in ``batch``
    numerical_covariates : list-like
        List of covariates in the model which are numerical, rather than
        categorical

    Returns
    -------
    corrected : pandas.DataFrame
        A (n_features, n_samples) dataframe of the batch-corrected data
    """
    
    # transform adata to data frame of the right shape and 
    if issparse(adata.X): 
        X = adata.X.A 
    else: 
        X = adata.X
    data_a = X.T
    data_a = np.log(data_a + 1)
    data = pd.DataFrame(data = data_a, index = adata.var_names,
                        columns= adata.obs_names)
    
    if isinstance(batch, str):
        batch = pd.Series(adata.obs[batch])

    if isinstance(numerical_covariates, str):
        numerical_covariates = [numerical_covariates]
    if numerical_covariates is None:
        numerical_covariates = []

    if model is not None and isinstance(model, pd.DataFrame):
        model["batch"] = list(batch)
    else:
        model = pd.DataFrame({'batch': batch})

    batch_items = model.groupby("batch").groups.items()
    batch_levels = [k for k, v in batch_items]
    batch_info = [v for k, v in batch_items]
    n_batch = len(batch_info)
    n_batches = np.array([len(v) for v in batch_info])
    n_array = float(sum(n_batches))

    # drop intercept
    drop_cols = [cname for cname, inter in  ((model == 1).all()).iteritems() if inter == True]
    drop_idxs = [list(model.columns).index(cdrop) for cdrop in drop_cols]
    model = model[[c for c in model.columns if not c in drop_cols]]
    numerical_covariates = [list(model.columns).index(c) if isinstance(c, str) else c
            for c in numerical_covariates if not c in drop_cols]

    design = design_mat(model, numerical_covariates, batch_levels)

    sys.stderr.write("Standardizing Data across genes.\n")
    B_hat = np.dot(np.dot(la.inv(np.dot(design.T, design)), design.T), data.T)
    grand_mean = np.dot((n_batches / n_array).T, B_hat[:n_batch,:])
    var_pooled = np.dot(((data - np.dot(design, B_hat).T)**2), np.ones((int(n_array), 1)) / int(n_array))

    print('Found {} genes with zero variance. Will be zero after tranformation.'.\
      format(np.sum(var_pooled == 0)))
    
    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, int(n_array))))
    tmp = np.array(design.copy())
    tmp[:,:n_batch] = 0
    stand_mean  += np.dot(tmp, B_hat).T

    # need to be a bit careful with the zero variance genes
    s_data = np.where(var_pooled == 0, 0, \
        ((data - stand_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, int(n_array))))))
    s_data = pd.DataFrame(s_data, index=data.index, columns=data.columns)

    sys.stderr.write("Fitting L/S model and finding priors\n")
    batch_design = design[design.columns[:n_batch]]
    gamma_hat = np.dot(np.dot(la.inv(np.dot(batch_design.T, batch_design)), batch_design.T), s_data.T)

    delta_hat = []

    for i, batch_idxs in enumerate(batch_info):
        #batches = [list(model.columns).index(b) for b in batches]
        delta_hat.append(s_data[batch_idxs].var(axis=1))

    gamma_bar = gamma_hat.mean(axis=1) 
    t2 = gamma_hat.var(axis=1)
   

    a_prior = list(map(aprior, delta_hat))
    b_prior = list(map(bprior, delta_hat))

    sys.stderr.write("Finding parametric adjustments\n")
    gamma_star, delta_star = [], []
    for i, batch_idxs in enumerate(batch_info):
        #print '18 20 22 28 29 31 32 33 35 40 46'
        #print batch_info[batch_id]

        temp = it_sol(s_data[batch_idxs], gamma_hat[i],
                     delta_hat[i], gamma_bar[i], t2[i], a_prior[i], b_prior[i])

        gamma_star.append(temp[0])
        delta_star.append(temp[1])

    sys.stdout.write("Adjusting data\n")
    bayesdata = s_data
    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)


    for j, batch_idxs in enumerate(batch_info):

        dsq = np.sqrt(delta_star[j,:])
        dsq = dsq.reshape((len(dsq), 1))
        denom =  np.dot(dsq, np.ones((1, n_batches[j])))
        numer = np.array(bayesdata[batch_idxs] - np.dot(batch_design.loc[batch_idxs], gamma_star).T)

        bayesdata[batch_idxs] = numer / denom
   
    vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
    bayesdata = bayesdata * np.dot(vpsq, np.ones((1, int(n_array)))) + stand_mean
    
    # put back into the adata object
    data_f = bayesdata.as_matrix().transpose()
    data_f = np.exp(data_f) - 1
    data_f = scipy.sparse.csc_matrix(data_f)
    adata.X = data_f
 
    #return bayesdata

def it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=0.0001):
    n = (1 - np.isnan(sdat)).sum(axis=1)
    g_old = g_hat.copy()
    d_old = d_hat.copy()

    change = 1
    count = 0
    while change > conv:
        #print g_hat.shape, g_bar.shape, t2.shape
        g_new = postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = ((sdat - np.dot(g_new.values.reshape((g_new.shape[0], 1)), np.ones((1, sdat.shape[1])))) ** 2).sum(axis=1)
        d_new = postvar(sum2, n, a, b)
       
        change = max((abs(g_new - g_old) / g_old).max(), (abs(d_new - d_old) / d_old).max())
        g_old = g_new #.copy()
        d_old = d_new #.copy()
        count = count + 1
    adjust = (g_new, d_new)
    return adjust 

    

def aprior(gamma_hat):
    m = gamma_hat.mean()
    s2 = gamma_hat.var()
    return (2 * s2 +m**2) / s2

def bprior(gamma_hat):
    m = gamma_hat.mean()
    s2 = gamma_hat.var()
    return (m*s2+m**3)/s2

def postmean(g_hat, g_bar, n, d_star, t2):
    return (t2*n*g_hat+d_star * g_bar) / (t2*n+d_star)

def postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)


if __name__ == "__main__":
    # NOTE: run this first to get the bladder batch stuff written to files.
    """
    source("http://bioconductor.org/biocLite.R")
    biocLite("sva")

	library("sva")
	options(stringsAsFactors=FALSE)

	library(bladderbatch)
	data(bladderdata)

	pheno = pData(bladderEset)
	# add fake age variable for numeric
	pheno$age = c(1:7, rep(1:10, 5))
        write.table(data.frame(cel=rownames(pheno), pheno), row.names=F, quote=F, sep="\t", file="bladder-pheno.txt")

	edata = exprs(bladderEset)
    write.table(edata, row.names=T, quote=F, sep="\t", file="bladder-expr.txt")
	# use dataframe instead of matrix
	mod = model.matrix(~as.factor(cancer) + age, data=pheno)
    t = Sys.time()
	cdata = ComBat(dat=edata, batch=as.factor(pheno$batch), mod=mod, numCov=match("age", colnames(mod)))
    print(Sys.time() - t)
    print(cdata[1:5, 1:5])
    write.table(cdata, row.names=True, quote=F, sep="\t", file="r-batch.txt")
    """

    pheno = pd.read_table('bladder-pheno.txt', index_col=0)
    dat = pd.read_table('bladder-expr.txt', index_col=0)

    mod = patsy.dmatrix("~ age + cancer", pheno, return_type="dataframe")
    import time
    t = time.time()
    ebat = combat(dat, pheno['batch'], mod, "age")
    sys.stdout.write("%.2f seconds\n" % (time.time() - t))

    sys.stdout.write(str(ebat.iloc[:5, :5]))

    ebat.to_csv("py-batch.txt", sep="\t")

    ebat = combat(dat, pheno['batch'], None)
