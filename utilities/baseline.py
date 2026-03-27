import math
import numpy as np 

from scipy import signal 
from scipy.spatial import ConvexHull 
from scipy.linalg import solveh_banded
from sklearn.linear_model import LinearRegression
from scipy.ndimage.morphology import binary_erosion

class Baseline(object):    
    
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
    

    def airpls_baseline(self, intensities, smoothness_param=100, max_iters=10,
                        conv_thresh=0.001, verbose=False):
        '''
        Baseline corr. using adaptive iteratively reweighted penalized least squares.
        Also known as airPLS, 2010.
        http://pubs.rsc.org/EN/content/articlehtml/2010/an/b922045c
        https://code.google.com/p/airpls/
        https://airpls.googlecode.com/svn/trunk/airPLS.py
        '''
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
        return baseline
    
    
    def als_baseline(self, intensities, asymmetry_param=0.05, smoothness_param=1e6,
                     max_iters=10, conv_thresh=1e-5, verbose=False):
        '''Perform asymmetric least squares baseline removal.
        * http://www.science.uva.nl/~hboelens/publications/draftpub/Eilers_2005.pdf
        smoothness_param: Relative importance of smoothness of the predicted response.
        asymmetry_param (p): if y > z, w = p, otherwise w = 1-p.
                             Setting p=1 is effectively a hinge loss.
        '''
        smoother = self.Smoother(intensities, smoothness_param, deriv_order=2)
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
    
    
    def iterative_threshold(self, intensities, num_stds=3):
        thresh = intensities.mean(axis=-1) + num_stds * intensities.std(axis=-1)
        old_mask = np.zeros_like(intensities, dtype=bool)
        mask = intensities >= np.array(thresh, copy=False)[..., None]
        while (mask != old_mask).any():
            below = np.ma.array(intensities, mask=mask)
            thresh = below.mean(axis=-1) + num_stds * below.std(axis=-1)
            old_mask = mask
            mask = intensities >= np.array(thresh, copy=False)[..., None]
        return ~mask
    
    
    def fabc_baseline(self, intensities, dilation_param=50, smoothness_param=1e3):
        '''Fully Automatic Baseline Correction, by Carlos Cobas (2006).
        http://www.sciencedirect.com/science/article/pii/S1090780706002266
        '''
        cwt = signal.cwt(intensities, signal.ricker, (dilation_param,))
        dY = cwt.ravel() ** 2

        is_baseline = self.iterative_threshold(dY)
        is_baseline[0] = True
        is_baseline[-1] = True

        smoother = self.Smoother(intensities, smoothness_param, deriv_order=1)
        return smoother.smooth(is_baseline)

    
    def median_baseline(self, intensities, window_size=501):
        '''Perform median filtering baseline removal.
        Window should be wider than FWHM of the peaks.
        "A Model-free Algorithm for the Removal of Baseline Artifacts" Friedrichs 1995
        '''
        # Ensure the window size is odd
        if window_size % 2 == 0:
            window_size += 1
        # Enable batch mode
        if intensities.ndim == 2:
            window_size = (1, window_size)
        return signal.medfilt(intensities, window_size)
    
    
    def polyfit_baseline(self, intensities, deg=5, max_it=None, tol=None):
        
        # for not repeating ourselves in `envelope`
        if deg is None: deg = 3
        if max_it is None: max_it = 100
        if tol is None: tol = 1e-3

        order = deg + 1
        coeffs = np.ones(order)

        # try to avoid numerical issues
        cond = math.pow(abs(intensities).max(), 1. / order)
        x = np.linspace(0., cond, intensities.size)
        base = intensities.copy()

        vander = np.vander(x, order)
        vander_pinv = np.linalg.pinv(vander)

        for _ in range(max_it):
            coeffs_new = np.dot(vander_pinv, intensities)

            if np.linalg.norm(coeffs_new - coeffs) / np.linalg.norm(coeffs) < tol:
                break

            coeffs = coeffs_new
            base = np.dot(vander, coeffs)
            intensities = np.minimum(intensities, base)

        return base
    
    
    
    def convex_baseline(self, intensities, iters=200):
        '''
        A novel baseline correction method using convex optimization 
        framework in laser-induced breakdown spectroscopy quantitative analysis.
        https://www.sciencedirect.com/science/article/abs/pii/S0584854716301975
        '''
        
        x = np.arange(0,intensities.shape[0])
    
        def rubberband(x, intensities):
            # Find the convex hull
            v = ConvexHull(list(zip(x, intensities))).vertices
            # Rotate convex hull vertices until they start from the lowest one
            v = np.roll(v, -v.argmin())
            # Leave only the ascending part
            v = v[:v.argmax()]
            # Create baseline using linear interpolation between vertices
            return np.interp(x, x[v], intensities[v])

        for _ in range(iters):
            ymax, ymin = intensities.max(), intensities.min()
            F = (ymax - ymin) / 10
            x0, xE = x.shape[0]//2, x.shape[0]
            f_x = F * (x - x0) ** 2 / (xE - x0) ** 2
            intensities = f_x + intensities
            intensities = intensities - rubberband(x, intensities)
        return intensities
    
    
    def rubberband_baseline(self,intensities, num_iters=8, num_ranges=64):
        '''Bruker OPUS method. If num_iters=0, uses basic method from OPUS.
        Method detailed in Pirzer et al., 2008 US Patent No. US7359815B2'''
        bands = np.arange(0, intensities.shape[0])
        def _rubberband(bands, intensities, num_ranges):
            '''Basic rubberband method,
            from p.77 of "IR and Raman Spectroscopy" (OPUS manual)'''
            # create n ranges of equal size in the spectrum
            range_size = len(intensities) // num_ranges
            y = intensities[:range_size * num_ranges].reshape((num_ranges, range_size))
            # find the smallest intensity point in each range
            idx = np.arange(num_ranges) * range_size + np.argmin(y, axis=1)
            # add in the start and end points as well, to avoid weird edge effects
            if idx[0] != 0:
                idx = np.append(0, idx)
            if idx[-1] != len(intensities) - 1:
                idx = np.append(idx, len(intensities) - 1)
            baseline_pts = np.column_stack((bands[idx], intensities[idx]))
            # wrap a rubber band around the baseline points
            hull = ConvexHull(baseline_pts)
            hidx = idx[hull.vertices]
            # take only the bottom side of the hull
            left = np.argmin(bands[hidx])
            right = np.argmax(bands[hidx])
            mask = np.ones(len(hidx), dtype=bool)
            for i in range(len(hidx)):
                if i > right and (i < left or right > left):
                    mask[i] = False
                elif i < left and i < right:
                    mask[i] = False
            hidx = hidx[mask]
            hidx = hidx[np.argsort(bands[hidx])]
            # interpolate a baseline
            return np.interp(bands, bands[hidx], intensities[hidx])

        y = intensities.copy()
        if num_iters > 0:
            for _ in range(num_iters):
                yrange = y.max() - y.min()
                x_center = bands[0]+(bands[-1] - bands[0]) / 2.
                tmp = (bands - x_center) ** 2
                y += yrange / 10. * tmp / tmp[-1]
                baseline = _rubberband(bands, y, num_ranges)
                y = y - baseline
            final_baseline = intensities - y
        else:
            final_baseline = _rubberband(bands, y, num_ranges)
        return final_baseline
    
    
    def poly(self, input_array_for_poly, degree_for_poly):
        '''qr factorization of a matrix. q` is orthonormal and `r` is upper-triangular.
        - QR decomposition is equivalent to Gram Schmidt orthogonalization, which builds a sequence of orthogonal polynomials that approximate your function with minimal least-squares error
        - in the next step, discard the first column from above matrix.
        - for each value in the range of polynomial, starting from index 0 of pollynomial range, (for k in range(p+1))
            create an array in such a way that elements of array are (original_individual_value)^polynomial_index (x**k)
        - concatenate all of these arrays created through loop, as a master array. This is done through (np.vstack)
        - transpose the master array, so that its more like a tabular form(np.transpose)'''
        input_array_for_poly = np.array(input_array_for_poly)
        X = np.transpose(np.vstack([input_array_for_poly**k for k in range(degree_for_poly+1)]))
        return np.linalg.qr(X)[0][:,1:] 

    
    def modpolyfit_baseline(self, intensities, degree=2,repitition=100,gradient=0.001):
        '''Implementation of Modified polyfit method from paper: Automated Method for Subtraction of Fluorescence 
        from Biological Raman Spectra, by Lieber & Mahadevan-Jansen (2003) 
        degree: Polynomial degree, default is 2
        repitition: How many iterations to run. Default is 100
        gradient: Gradient for polynomial loss, default is 0.001. It measures incremental gain over each iteration. 
        If gain in any iteration is less than this, further improvement will stop.
        '''
        input_array=intensities
        lin=LinearRegression()
        criteria=np.inf

        baseline=[]
        corrected=[]

        ywork=input_array
        yold=input_array
        yorig=input_array

        polx=self.poly(list(range(1,len(yorig)+1)),degree)
        nrep=0

        while (criteria>=gradient) and (nrep<=repitition):
            ypred=lin.fit(polx,yold).predict(polx)
            ywork=np.array(np.minimum(yorig,ypred))
            criteria=sum(np.abs((ywork-yold)/yold))
            yold=ywork
            nrep+=1
        return ypred


    def imodpolyfit_baseline(self, intensities, degree=2,repitition=100,gradient=0.001):
        '''IModPoly from paper: Automated Autofluorescence Background Subtraction Algorithm for Biomedical Raman Spectroscopy, 
        by Zhao, Jianhua, Lui, Harvey, McLean, David I., Zeng, Haishan (2007).
        degree: Polynomial degree, default is 2.
        repitition: How many iterations to run. Default is 100
        gradient: Gradient for polynomial loss, default is 0.001. It measures incremental gain over each iteration. 
        If gain in any iteration is less than this, further improvement will stop.
        '''
        lin=LinearRegression()
        yold=np.array(intensities)
        yorig=np.array(intensities) 
        corrected=[]

        nrep=1
        ngradient=1

        polx=self.poly(list(range(1,len(yorig)+1)),degree)
        ypred=lin.fit(polx,yold).predict(polx)
        Previous_Dev=np.std(yorig-ypred)

        #iteration1
        yold=yold[yorig<=(ypred+Previous_Dev)]
        polx_updated=polx[yorig<=(ypred+Previous_Dev)]
        ypred=ypred[yorig<=(ypred+Previous_Dev)]

        for i in range(2,repitition+1):
            if i>2:
                Previous_Dev=DEV
            ypred=lin.fit(polx_updated,yold).predict(polx_updated)
            DEV=np.std(yold-ypred)

            if np.abs((DEV-Previous_Dev)/DEV) < gradient:
                break
            else:
                for i in range(len(yold)):
                    if yold[i]>=ypred[i]+DEV:
                        yold[i]=ypred[i]+DEV
        baseline=lin.predict(polx)
        return baseline


    def dietrich_baseline(self, intensities, half_window=16, num_erosions=10):
        '''
        Fast and precise automatic baseline correction of ... NMR spectra, 1991.
        http://www.sciencedirect.com/science/article/pii/002223649190402F
        http://www.inmr.net/articles/AutomaticBaseline.html
        '''
        bands = np.arange(0, intensities.shape[0])
        # Step 1: moving-window smoothing
        w = half_window * 2 + 1
        window = np.ones(w) / float(w)
        Y = intensities.copy()
        if Y.ndim == 2:
            window = window[None]
        Y[..., half_window:-half_window] = signal.convolve(Y, window, mode='valid')

        # Step 2: Derivative.
        dY = np.diff(Y) ** 2

        # Step 3: Iterative thresholding.
        is_baseline = np.ones(Y.shape, dtype=bool)
        is_baseline[..., 1:] = self.iterative_threshold(dY)

        # Step 3: Binary erosion, to get rid of peak-tops.
        mask = np.zeros_like(is_baseline)
        mask[..., half_window:-half_window] = True
        s = np.ones(3, dtype=bool)
        if Y.ndim == 2:
            s = s[None]
        is_baseline = binary_erosion(is_baseline, structure=s,
                                     iterations=num_erosions, mask=mask)

        # Step 4: Reconstruct baseline via interpolation.
        if Y.ndim == 2:
            return np.row_stack([np.interp(bands, bands[m], y[m])
                                 for y, m in zip(intensities, is_baseline)])
        return np.interp(bands, bands[is_baseline], intensities[is_baseline])
    

    def kajfosz_kwiatek_baseline(self, intensities, top_width=0,bottom_width=50, exponent=2,tangent=False):
        """
        This function uses an enhanced version of the algorithm published by
        Kajfosz, J. and Kwiatek, W.M. (1987)  "Non-polynomial approximation of
        background in x-ray spectra." Nucl. Instrum. Methods B22, 78-81.
        top_width:
          Specifies the width of the polynomials which are concave upward.
          The top_width is the full width in energy units at which the
          magnitude of the polynomial is 0.1 of max. The default is 0, which
          means that concave upward polynomials are not used.
        bottom_width:
          Specifies the width of the polynomials which are concave downward.
          The bottom_width is the full width in energy units at which the
          magnitude of the polynomial is 0.1 of max. The default is 50.
        exponent:
          Specifies the power of polynomial which is used. The power must be
          an integer. The default is 2, i.e. parabolas. Higher exponents,
          for example EXPONENT=4, results in polynomials with flatter tops
          and steeper sides, which can better fit spectra with steeply
          sloping backgrounds.
        tangent:
          Specifies that the polynomials are to be tangent to the slope of the
          spectrum. The default is vertical polynomials. This option works
          best on steeply sloping spectra. It has trouble in spectra with
          big peaks because the polynomials are very tilted up inside the
          peaks.
        For more info, see:
        cars9.uchicago.edu/software/idl/mca_utility_routines.html#FIT_BACKGROUND
        """
        
        def _kk_lookup_table(spectrum, width, exponent, slope, ref_ampl):
            nchans = len(spectrum)
            chan_width = width / (2. * slope)
            denom = chan_width ** exponent
            indices = np.arange(-nchans, nchans + 1)
            power_funct = indices ** exponent * (ref_ampl / denom)
            power_funct = power_funct[power_funct <= 1]
            max_index = len(power_funct) // 2 - 1
            return power_funct, max_index
        
        bands = np.arange(0, intensities.shape[0])
        REFERENCE_AMPL = 0.1
        MAX_TANGENT = 2

        nchans = len(intensities)
        # Normalize intensities for widths to make sense.
        scale_factor = intensities.max()
        scratch = intensities / scale_factor
        slope = abs(np.diff(bands).mean())

        # Fit functions which come down from top
        if top_width > 0:
            power_funct, max_index = _kk_lookup_table(
                scratch, top_width, exponent, slope, REFERENCE_AMPL)

            bckgnd = scratch.copy()
            for center_chan in range(nchans):
                first_chan = max((center_chan - max_index), 0)
                last_chan = min(center_chan + max_index + 1, nchans)
                f = first_chan - center_chan + max_index
                l = last_chan - center_chan + max_index
                lin_offset = scratch[center_chan]
                new_bckgnd = power_funct[f:l] + lin_offset
                old_bckgnd = bckgnd[first_chan:last_chan]
                np.copyto(old_bckgnd, new_bckgnd, where=(new_bckgnd > old_bckgnd))

            # Copy this approximation of background to scratch
            scratch = bckgnd.copy()
        else:
            bckgnd = np.empty_like(scratch)

        # Fit functions which come up from below
        power_funct, max_index = _kk_lookup_table(
            scratch, bottom_width, exponent, slope, REFERENCE_AMPL)

        bckgnd.fill(-np.inf)
        for center_chan in range(nchans - 1):
            if tangent:
                # Find slope of tangent to spectrum at this channel
                first_chan = max(center_chan - MAX_TANGENT, 0)
                last_chan = min(center_chan + MAX_TANGENT + 1, nchans)
                denom = center_chan - np.arange(last_chan - first_chan)
                tangent_slope = (scratch[center_chan] -
                                 scratch[first_chan:last_chan]) / np.maximum(denom, 1)
                tangent_slope = np.sum(tangent_slope) / (last_chan - first_chan)

            first_chan = max(center_chan - max_index, 0)
            last_chan = min(center_chan + max_index + 1, nchans)
            lin_offset = scratch[center_chan]
            if tangent:
                nc = last_chan - first_chan
                lin_offset += (np.arange(nc) - nc / 2.) * tangent_slope

            # Find the maximum height of a function centered on this channel
            # such that it is never higher than the counts in any channel
            f = first_chan - center_chan + max_index
            l = last_chan - center_chan + max_index
            pf = power_funct[f:l] - lin_offset
            height = (scratch[first_chan:last_chan] + pf).min()

            # We now have the function height. Set the background to the
            # height of the maximum function amplitude at each channel
            new_bckgnd = height - pf
            old_bckgnd = bckgnd[first_chan:last_chan]
            np.copyto(old_bckgnd, new_bckgnd, where=(new_bckgnd > old_bckgnd))

        return bckgnd * scale_factor


def butterworth_filter(dat):
    # Butterworth Filter
    N = 3  # Filter order
    Wn = 0.05  # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')
    return signal.filtfilt(B, A, dat)

def Smoothing(df):
    for i in range(len(df)):
        df.iloc[i]= butterworth_filter(np.array(df.iloc[i]))
    return df
    
def Baseline_Reduction(df):
    baseline = Baseline()
    for i in range(len(df)):
        baseline_temp = baseline.als_baseline(df.iloc[i].values.ravel())
        df.iloc[i]= df.iloc[i]-baseline_temp
    return df