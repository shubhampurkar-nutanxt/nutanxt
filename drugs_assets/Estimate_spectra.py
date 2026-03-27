
import numpy as np
from scipy import signal
from scipy.interpolate import splev, splrep
from scipy.linalg import solveh_banded
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial import ConvexHull
from sklearn.metrics.pairwise import (cosine_similarity, euclidean_distances,
                                      manhattan_distances)
from scipy.stats import pearsonr
import math
import numpy as np 
from scipy import signal 
from scipy.spatial import ConvexHull 
from scipy.linalg import solveh_banded
from sklearn.linear_model import LinearRegression
from scipy.ndimage.morphology import binary_erosion
from sklearn.metrics.pairwise import cosine_similarity,manhattan_distances,euclidean_distances

import warnings
warnings.filterwarnings('ignore')
import math
import numpy as np 

from scipy import signal 
from scipy.spatial import ConvexHull 
from scipy.linalg import solveh_banded
from sklearn.linear_model import LinearRegression
from scipy.ndimage.morphology import binary_erosion

class Smoother(object):
    
        def __init__(self, Y, smoothness_param, deriv_order=1):
            self.y = Y
            assert deriv_order > 0, 'deriv_order must be an int > 0'
            d = np.zeros(deriv_order * 2 + 1, dtype=int)
            d[deriv_order] = 1
            d = np.diff(d, n=deriv_order)
            n = self.y.shape[0]
            k = len(d)
            s = float(smoothness_param)
            diag_sums = np.vstack([
                np.pad(s * np.cumsum(d[-i:] * d[:i]), ((k - i, 0),), 'constant')
                for i in range(1, k + 1)])
            upper_bands = np.tile(diag_sums[:, -1:], n)
            upper_bands[:, :k] = diag_sums
            for i, ds in enumerate(diag_sums):
                upper_bands[i, -i - 1:] = ds[::-1][:i + 1]
            self.upper_bands = upper_bands

        def smooth(self, w):
            foo = self.upper_bands.copy()
            foo[-1] += w  # last row is the diagonal
            return solveh_banded(foo, w * self.y, overwrite_ab=True, overwrite_b=True)

    
    
def airpls_baseline(intensities, smoothness_param=100, max_iters=10,
                        conv_thresh=0.001, verbose=False):
        '''
        Baseline corr. using adaptive iteratively reweighted penalized least squares.
        Also known as airPLS, 2010.
        http://pubs.rsc.org/EN/content/articlehtml/2010/an/b922045c
        https://code.google.com/p/airpls/
        https://airpls.googlecode.com/svn/trunk/airPLS.py
        '''
        smoother = Smoother(intensities, smoothness_param)
        total_intensity = np.abs(intensities).sum()
        w = np.ones(intensities.shape[0])

        for i in range(1, int(max_iters + 1)):
            baseline = smoother.smooth(w)
            # Compute error (sum of distances below the baseline).
            corrected = intensities - baseline
            mask = corrected < 0
            baseline_error = -corrected[mask]
            total_error = baseline_error.sum()
            # Check convergence as a fraction of total intensity.
            conv = total_error / total_intensity
            if verbose:
                print(i, conv)
            if conv < conv_thresh:
                break
            # Set peak weights to zero.
            w[~mask] = 0
            # Set baseline weights.
            baseline_error = baseline_error / total_error
            w[mask] = np.exp(i * baseline_error)
            w[0] = np.exp(i * baseline_error.min())
            w[-1] = w[0]
        else:
            print('airPLS did not converge in %d iterations' % max_iters)
        return baseline


def estimate_structure(data,**kwargs):
    """
    data -> list [array] eg [ibu.list()]
    ratio - > list [float] if 12.5 == 0.125
    
    """
    ratio = 0
    
    def processed_baseline(frame):
        """
        take frame 
        processed baseline alignment and obtain the norm graph after baselined
        """
        obtained,base = [],[]
        for i in frame:base.append(baseline_mine(i['data'])*i["ratio"]) # compute baseline output of each sample      
        for i in np.array(base).transpose(): obtained.append(max(i)) # merge the data 
        return obtained
    def get_baseline(x):
        """
        return just baseline
        
        """
        x = np.array(x)
        x = airpls_baseline(x,smoothness_param=10)
        x = norm(x)
        return x


    def baseline_mine(x):
        """
        return baseline adjusted
        """
        x = np.array(x)
        x-= airpls_baseline(x,smoothness_param=10)
        x = np.where(x>0,x,0)
        x = norm(x)
        return x
    
    def norm(x):
        """
        Norm the incoming
        """
        x =  np.array(x)
        x/=(x.max()-x.min())
        return x
    
    params = kwargs.keys()

    num_of_drug = len(data)
    
    try: 
        assert(np.all(np.array([len(kwargs[k])==num_of_drug for k in params ]))==True)
        assert("ratio" in kwargs.keys())
        if len(kwargs['ratio']) in [0,1]: ratio =  1/num_of_drug
    except AssertionError: 
        ratio =  1/num_of_drug
        #else: raise ValueError("Ratio is not even please provide ratio properly")
    frame = []
    for i in range(num_of_drug): 
        d={};d[f"data"]=data[i]; d[f'ratio']= kwargs['ratio'][i] if ratio==0 else ratio
        frame.append(d)
        
    frame = sorted(frame,key=lambda x: x['ratio'])   
    obtained = processed_baseline(frame)     
    baseline_of_majority = get_baseline(frame[-1]['data'])   
    estimate = baseline_of_majority + obtained  
    estimate = norm(estimate)
    return estimate
