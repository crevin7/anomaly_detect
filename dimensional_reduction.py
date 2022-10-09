from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import numpy as np

def n_sampled_feats(X: np.ndarray, explained_var_ratio: float = 0.8) -> int:
    

    full_pca = PCA(n_components=None)
    full_pca.fit(X)
    feat_var = full_pca.explained_variance_ratio_
    assert np.sum(feat_var) == 1
    sum_var = 0
    
    for en, var in enumerate(feat_var):
        sum_var += var
        if sum_var > explained_var_ratio:
            return en+1
    return


def get_pca_repres(X: np.ndarray, explained_var_ratio: float = 0.8) -> np.ndarray:

    n_components = n_sampled_feats(X, explained_var_ratio)
    pca = PCA(n_components=n_components)
    feats = pca.fit_transform(X)
    assert np.sum(pca.explained_variance_ratio_) > 0.8, np.sum(pca.explained_variance_ratio_)
    return feats

def get_explained_var_ratio(X: np.ndarray, feats: np.ndarray) -> float:

    linreg = LinearRegression(fit_intercept = False)
    linreg.fit(X, feats)
    lin_rec = linreg.predict(X)

    sq_frob_norm = np.linalg.norm(lin_rec - X)**2
    ecplained_var_ratio = 1 - sq_frob_norm/np.linalg.norm(X)
    return ecplained_var_ratio
