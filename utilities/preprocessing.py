import numpy as np 
import pandas as pd
import peakutils
from scipy import signal 
from scipy.optimize import curve_fit
from scipy.linalg import solveh_banded
from sklearn.preprocessing import MinMaxScaler


class Preprocessing:
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
    
    def __init__(self,dataset=None,peak=None):
        self.dataset = dataset
        self.database = {11:"Lidocane",
                        8:"Vancomycin"}
        self.peak = peak
    
    def baseline(self):
        self.dataset -= self.dataset.apply(lambda x: peakutils.baseline(x))
        return self.dataset
    
    def singleBL2(self, smoothness_param=100, max_iters=10,
                        conv_thresh=0.001, verbose=False):
        '''
        Baseline corr. using adaptive iteratively reweighted penalized least squares.
        Also known as airPLS, 2010.
        http://pubs.rsc.org/EN/content/articlehtml/2010/an/b922045c
        https://code.google.com/p/airpls/
        https://airpls.googlecode.com/svn/trunk/airPLS.py
        '''
        intensities = self.database
        smoother = self.Smoother(intensities, smoothness_param)
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
        return intensities-baseline
    
    def singleBL(self,intensities=None, asymmetry_param=0.05, smoothness_param=1e6,
                     max_iters=10, conv_thresh=1e-5, verbose=False):
        '''Perform asymmetric least squares baseline removal.
        * http://www.science.uva.nl/~hboelens/publications/draftpub/Eilers_2005.pdf
        smoothness_param: Relative importance of smoothness of the predicted response.
        asymmetry_param (p): if y > z, w = p, otherwise w = 1-p.
                             Setting p=1 is effectively a hinge loss.
        '''
        if intensities is None:
            intensities = self.dataset
        smoother = self.Smoother(intensities, smoothness_param, deriv_order=2)
#         smoother = self.butterworth_filter(intensities)
        # Rename p for concision.
        p = asymmetry_param
        # Initialize weights.
        w = np.ones(intensities.shape[0])
        for i in range(max_iters):
            z = smoother.smooth(w)
            mask = intensities > z
            new_w = p * mask + (1 - p) * (~mask)
            conv = np.linalg.norm(new_w - w)
            if verbose:
                print(i + 1), conv
            if conv < conv_thresh:
                break
            w = new_w
        else:
            print('ALS did not converge in %d iterations' % max_iters)
        return self.butterworth_filter(intensities-z)
    
    def singleBL_(self,dataset=None,peak=None,buffer=None,smooth = True,inplace=False):
        if peak is None:
            peak = self.peak
        if dataset is None:
            k = self.dataset.T
        else:
            k = dataset
        k.index = k.index.astype('int')
        base = k.apply(lambda x:peakutils.baseline(x))

        newdata = k - base
        if buffer is not None:
            
            newdata1 = newdata.loc[peak-buffer:peak+buffer]
        else:
            newdata1 = newdata
        if smooth:
            data2 = newdata1.apply(lambda x: self.butterworth_filter(x))
        else:
            data2 = newdata1
        data2 = data2.T
        for row in range(data2.shape[0]):
            data2.iloc[row] = data2.iloc[row].apply(lambda x:  0 if x < 0 else x)
        if inplace:
            self.dataset = data2
        return data2 
    def butterworth_filter(self,dat):
        # Butterworth Filter
        N = 3  # Filter order
        Wn = 0.05  # Cutoff frequency
        B, A = signal.butter(N, Wn, output='ba')
        return pd.Series(signal.filtfilt(B, A, dat))
    def Outlier_removal(self,dataset=None,peak=None,factor=1.5,left=25,right=75):
        if peak is None:
            peak = self.peak
        if dataset is None:
            dataset = self.dataset
            peak = dataset.shape[0] // 2
        def iqr_outlier(x,factor,left=25,right=75):
            q1 = np.percentile(x,left, interpolation = 'midpoint')
            q3 = np.percentile(x,right, interpolation = 'midpoint')
            iqr = q3 - q1
            min_ = q1 - factor * iqr
            max_ = q3 + factor * iqr
            result_ = pd.Series([0] * len(x))
            result_[((x < min_) | (x > max_))] = 1
            return result_
        row1 = (np.array(dataset.iloc[:,peak]))
        done = dataset.loc[np.array(iqr_outlier(row1,factor,left,right)) == 0]
        return done 
    def doubleBL(self,peak=None,buffer=None,smooth = True):
        fst = self.singleBL(self.dataset,peak,buffer,smooth)
        snd = self.singleBL(fst,peak,buffer,smooth)
        return snd
    # def drugIdentifier(y):
    #     temp = np.mean(self.singleBL(dataset=y))
    #     indexes = peakutils.indexes(temp, thres=0.2, min_dist=30)
        # pyplot.figure(figsize=(10,6))
        # pplot(np.arange(350,350+temp.shape[0]), temp, indexes)
    def BlOut(self,peak=None,buffer=None,factor=0.01,visuals=False):
        temp = self.singleBL(peak=peak,buffer=buffer)
        out = self.Outlier_removal(temp,factor=factor)
        # if visuals:
        #     plt.plot(temp.T,color='red')
        #     plt.plot(out.T,color='green')
        return out,temp
    @staticmethod
    def quadratic(x, a, b, c):
        """X is the wavenumber"""
        return a*x**2 + b*x + c
    @staticmethod
    def cubic(x,a,b,c,d):
        return a*x**3 + b*x**2 + c*x + d
    @staticmethod
    def linear(x,a,b):
        return a*x + b  
    def waterNorm(self,target_peaks,zeroPer,data=None):
        water_df = zeroPer
        if data is None:
            
            test_df = pd.DataFrame({"drug":self.dataset,"Water":water_df}).T  
        else:
            test_df = pd.DataFrame({"drug":data,"Water":water_df}).T 
        test_n = test_df[[np.round(i,1) for i in target_peaks]]
        coord,_ = curve_fit(Preprocessing().quadratic,test_n.iloc[0],test_n.iloc[1])
        y_vals = [Preprocessing().quadratic(i,*coord) for i in test_df.iloc[0].values] 
        return np.array(y_vals)
    def waterPlusSingleBL(self,tp,zeroPer,data=None):
        if data is None:
            data = self.dataset
        return self.singleBL(self.waterNorm(tp,zeroPer,data))
    def fine_xcal(self, reference,df=None):
        """
        df : DataFrame composed of pre x-calibrated sample scan
        reference : list of reference points for provided sample
        returns fine x-calibrated sample spectrum as DataFrame
        """
        if df is None:
            df = pd.DataFrame(self.dataset).T
        data=[]
        size = len(df.columns)

        peaks = [i for i in reference if i<size]
        window = min([abs(peaks[i]-peaks[i+1]) for i in range(len(peaks)-1)])//2
    #     print(peaks)
    #     print(window)
        measured = [df.iloc[:,peak-window:peak+window].idxmax(axis=1)[0] for peak in peaks]
        diff = [(i-j)for i,j in zip(peaks,measured)]
        dips=[df.iloc[:,peaks[i]:peaks[i+1]].idxmin(axis=1)[0] for i in range(len(peaks)-1)]
        dips.append(df.columns[0])
        dips.append(df.columns[-1])
        dips.sort()

        if diff[0]>0 : data = data + [df.iloc[0,0]]*diff[0]
        for i in range(len(peaks)):data = data + list(df.iloc[:,dips[i] - diff[i]:dips[i+1] - diff[i]-1].values[0])
        if len(data) < size : data = data + [data[-1]]*(size-len(data))
        else : data = data[:size]
    #     dft = pd.DataFrame(data, index=np.arange(200,size+200), columns=['0']).T
        dft = pd.DataFrame(data, index=np.arange(0,size), columns=['0']).T
        return dft
    

def butterworth_filter(dat):
        # Butterworth Filter
        N = 3  # Filter order
        Wn = 0.05  # Cutoff frequency
        B, A = signal.butter(N, Wn, output='ba')
        return signal.filtfilt(B, A, dat)

def als_baseline(intensities, asymmetry_param=0.05, smoothness_param=1e6,
                     max_iters=10, conv_thresh=1e-5, verbose=False):
        '''Perform asymmetric least squares baseline removal.
        * http://www.science.uva.nl/~hboelens/publications/draftpub/Eilers_2005.pdf
        smoothness_param: Relative importance of smoothness of the predicted response.
        asymmetry_param (p): if y > z, w = p, otherwise w = 1-p.
                             Setting p=1 is effectively a hinge loss.
        '''
        smoother = Smoother(intensities, smoothness_param, deriv_order=2)
        # Rename p for concision.
        p = asymmetry_param
        # Initialize weights.
        w = np.ones(intensities.shape[0])
        for i in range(max_iters):
            z = smoother.smooth(w)
            mask = intensities > z
            new_w = p * mask + (1 - p) * (~mask)
            conv = np.linalg.norm(new_w - w)
            if verbose:
                print(i + 1), conv
            if conv < conv_thresh:
                break
            w = new_w
        else:
            print('ALS did not converge in %d iterations' % max_iters)
        return z
        
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


def normalize_minmax(df):
    data = df.T
    scaler = MinMaxScaler()
    scaler.fit(data)
    normalize=pd.DataFrame(scaler.transform(data)).T
    return normalize