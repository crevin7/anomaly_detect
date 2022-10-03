import scipy.stats as stats
import pingouin as pg
import statsmodels.api as sm
import numpy as np
from typing import Dict, Sequence, Callable, Tuple
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression



def double_median_abs_deviation(points: np.ndarray,
                                median_fun: Callable[np.ndarray, float] = np.median,
                                double_mad: bool = True,
                                check_scaled: bool = True
                               ) -> Tuple[float, float, float]:
    """Compute (double) Median Absolute Deviation MAD and median

    Parameters
    ----------
    points
        sample points from a (non-gaussian) distribution
    median_fun
        function utilised to compute the median
    double_mad
        if True compute the double mad
    check_scaled
        if True check if points are scaled
    
    Returns
    -------
    left MAD, right MAD, median
    """ 
    if check_scaled:
        assert np.mean(points) < 1e-2
        assert abs(np.std(points) - 1) < 1e-2, f"{np.std(points)}"
    median = median_fun(points)
        
    if double_mad:
        return b*abs(median_fun(points[points<median] - median)), b*abs(median_fun(points[points>=median] - median)), median
    
    return abs(median_fun(points - median)), abs(median_fun(points - median)), median


def is_mad_outlier(point: float,
                     mad: Union[float, Tuple[float, float]],
                     k: float = 1.4826,
                      median: float, co_factor: float = 3) -> bool:
    """ check if point is an outlier according to MAD

    Parameters
    ----------
    point
        point to evaluate if outlier
    mad
        MAD value(s)
    k
        normalising factor (k*MAD = standard deviation)
    median
        sample median of the distribution
    co_factor
        cut-off factor, defining the (left and right) outlier cutoff
    
    Returns
    -------
    True if point is outlier
    """
    if isinstance(mad, float):
        mad = (mad, mad)
        
    return point > median + k*mad[1]*co_factor or point < median - k*mad[0]*co_factor

