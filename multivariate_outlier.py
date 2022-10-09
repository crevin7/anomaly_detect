from KDEpy import FFTKDE
import numpy as np
import logging

SMALL_DATA_POINTS = 1000
def is_outlier(likelihood_score: np.ndarray, threshold: float = 0.005) -> bool:

    return likelihood_score<threshold

def get_kde_likelihodd(data: np.ndarray, kernel: str ='gaussian', bandwith: str = 'ISJ') -> np.ndarray:
    data_points = len(data)
    if data_points< SMALL_DATA_POINTS:
        logging.warning(f'Data contains only {data_points}, better to use Silverman bandwith')
    
    _, likelihood = FFTKDE(kernel=kernel, bw=bandwith).fit(data).evaluate()

    return likelihood



