import math
import numpy as np


def normalize_score(score: float, pct_distr_list: list) -> float:
    '''
    Created on the assumption that KNN cosine similarity between speaker embeddings and topic embeddings adhered to a corpus-wide
    distribution of distances. Now I'm seeing that any given speaker in any given episode has a narrow range of cosine distances to
    most topics. I'd like to think the topic rankings for a given speaker episode still have meaning, but often the lowest-scoring 
    topic for one speaker episode exceeds the highest-scoring topic for another speaker in the same episode. And even when this 
    function is applied the differences in 'normalized' distances between the same speaker episode and multiple topics are miniscule.
    '''
    low = 0
    high = len(pct_distr_list)-1
    percentile = -1
    while low <= high:
        mid = math.floor((high + low) / 2)
        if score >= pct_distr_list[mid]:
            if mid == len(pct_distr_list)-1:
                return float(mid)
            elif score < pct_distr_list[mid+1]:
                percentile = mid
                break
            else:
                low = mid+1
        elif mid == 0 and score < pct_distr_list[mid]:
            return 0.0
        else:
            high = mid-1
    
    if percentile == -1:
        raise(f'Failure to normalize score={score} within percent distribution={pct_distr_list}')
   
    percentile_decimal = (score - pct_distr_list[percentile]) / (pct_distr_list[percentile+1] - pct_distr_list[percentile])
    percentile += percentile_decimal

    return round(percentile, 2)


def scale_values(values: list, low: int = 0, high: int = 1) -> list:
    raw_low = np.min(values)
    raw_high = np.max(values)
    raw_range = raw_high - raw_low
    scaled_range = high - low
    scaled_values = []
    for v in values:
        scaled_v = (v - raw_low) / raw_range * scaled_range + low
        scaled_values.append(scaled_v)
    return scaled_values
