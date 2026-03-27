import json
import os

import numpy as np
import pandas as pd
import math
import sklearn.linear_model as linear_model
from scipy.signal import find_peaks,peak_prominences
from scipy.stats import pearsonr
from scipy.linalg import solveh_banded
from sklearn.metrics.pairwise import (cosine_similarity, euclidean_distances,
                                      manhattan_distances)
from sklearn.neighbors import KNeighborsRegressor
from utilities.preprocessing import (airpls_baseline, als_baseline,
                                     butterworth_filter)
from scipy.optimize import curve_fit
import pickle
import traceback
# from config import DIR_MODELS
from drugs_assets.oak_database_etl  import Q_DRUG,IDENTIFY_DRUG,NARCO_DRUG,NON_NARCO_DRUG,database_drugs,DRUGSLIST,SYRNAME,ID2_DB_DATA
from drugs_assets.Estimate_spectra import estimate_structure
from itertools import combinations_with_replacement,combinations
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
#print(database_drugs['Ibuprofen'])
print(DRUGSLIST)
print(database_drugs.keys())
#####################Global Variable########################
version =  "2.0.2 possible match removed"
# version =  "2.0.1 syringe identification removed"
print(f"Loaded Identification version - > {version}")

size = 4
prominence = 0.03

NARCO_DRUGS_LIST = ["Heroin","Methamphetamine","Cocaine","Mdma","Methanol","Cocainehcl","Sersheroin"]
###################### Dleaning_File ########################
dlearning_file = os.path.join(os.path.dirname(__file__),'drugs_assets/dlearning.json')

############### Utilities of current module ###############
def pre_check_intesity(spectra):
    global prominence
    def number_of_peaks(spectra):
        processed = preprocess(spectra)
        return find_peaks(processed,prominence=prominence)[0].shape[0]
    no_of_peaks = number_of_peaks(spectra=spectra)
    print("Number of peaks found = ",no_of_peaks)
    spectra = spectra[0:1600]
    if max(spectra)>5000:return True
    else: return False

def flourencense_identify(x,just_peaks=False):
    global prominence
    def norm(x):
        x = np.array(x)
        return (x-x.min())/(x.max()-x.min())
    def peak_q(y):
        y = preprocess(y).values
        y = y[200:1500]
        peaks, properties = find_peaks(y,prominence=prominence, height=0.05)
        peak_quality = np.sum(properties['peak_heights'])
        return round(peak_quality,3)
    def auc(x):
        x = norm(x)
        auc_val_full =  np.trapz(x[200:1500])
        auc_val_limit  = np.trapz(x[200:1000])
        return auc_val_full,auc_val_limit
    def peaks_count(x):
        x = norm(x)[200:1500]
        x  = find_peaks(x,prominence=prominence)[0].shape[0]
        return x  
    if just_peaks:
        def number_of_peaks(spectra):
            processed = preprocess(spectra)
            return find_peaks(processed,prominence=prominence)[0].shape[0]
        no_of_peaks = number_of_peaks(spectra=x)
        if  no_of_peaks>60: return True
    if len(x)>0:
        auc_val1,auc_val2 = auc(norm(x))
        pc = peaks_count(x)
        #print(f"Peaks Count: {pc}, Peaks Quality => {peak_q(x)}, AUC = {auc_val1},{auc_val2}")
        return (pc<3) or (peak_q(x)<1) or (auc_val1>1200) or (auc_val2>750)
    return False

def peak_location_proba(test):
    d= {}
    for i in IDENTIFY_DRUG:
        d[i]=peak_location_score(test,i)
    d = sorted(d.items(), key=lambda kv: kv[1],reverse=True)
    return d

def check_narco_peak(x):
    if len(x)>0:
        return max(x)>0.1
    else: return False
    
def get_ref_peaks(ref,drugname):
    global prominence
    ref = preprocess(ref)
    g_peaks = database_drugs[drugname]['peaks_list']
    pickup = 200 if len(ref) else 350
    ref_peaks = find_all_near_peaks(find_peaks(ref,prominence=prominence)[0]+pickup,g_peaks)
    return ref_peaks
    
def find_nearest(array, value:int,multi=False):  ### 
        '''
        Find nearest peak
        by comparing two list
        '''
        psize = 11 ####
        array = np.asarray(array)
        if multi:
            idx = (np.abs(array - value)).argsort()[:3]
            val = [i  for i in array[idx] if abs(i-value)<psize]
            return val
        else: 
            idx = (np.abs(array - value)).argmin()
            if np.abs(array[idx]-value)<psize:
                return array[idx]
            else: 
                return 0
    
def find_all_near_peaks(peaks,gpeaks,multi=False,strict_mode=False):
    def strict_peak_catch(d):
        kl,k2 = {},{}
        for i,j in d.items():
            for f  in j:
                try: kl[f].append(i)
                except: kl[f]=[i]
        for i,j in kl.items():
            if len(j)>1:
                arg = np.abs(np.array(j)-i).argmin()
                kl[i]=[j[arg]]
        for i,j in d.items():
            k2[i]=[]
            for u in j:
                if kl[u][0]==i:
                    try: k2[i].append(u)
                    except: k2[i]=[u]
        return k2
    d={}
    for p in gpeaks:
        d[p]=find_nearest(peaks,p,multi=multi)
    if strict_mode:
            d = strict_peak_catch(d)   
    return d

def dynamic_score(p,mode="shift"):
    def quar_func(k, a, b, c):
                return (a*(k**2)) + (k*b) +c
    if mode=='shift':
        if isinstance(p,int):
            x = [0,1,2,3,4,5,6,7,8,9]
            y = [1,1,1,1,1,0.99,0.985,0.97,0.96,0.95]
            popt, pcov = curve_fit(quar_func, x, y)
            x_test = np.arange(0,p+2)
            data = quar_func(x_test,*popt)
            return round(data[list(x_test).index(p)],2)
        
        else: raise ValueError("P must be integer for shift mode")
    if mode=='intensity':
        x =  [0,0.1,0.15,0.2,0.25,0.3]
        y = [1,0.9,0.85,0.8,0.75,0.7]
        popt, pcov = curve_fit(quar_func, x, y)
        x_test = np.arange(0,p+0.10,0.01)
        data = quar_func(x_test,*popt)
        x_test = list(map(lambda x: round(x,2),x_test))
    return round(data[list(x_test).index(p)],2)


def extract_peaks_from_database(database):
    peaks = {}
    for i in database.keys(): 
        dpeaks = database[i]['peaks_list']
        if len(dpeaks)>1:
            peaks[i]= dpeaks
    return peaks 

def get_range_of_peaks(drugname:str):
    peaks = database_drugs[drugname]['peaks_list']
    end = max(peaks)+100
    start = min(peaks)-100
    return (round(start,-1),round(end,-1))
    

def number_of_peaks(peaks,peaksref):
    global size
    def find_nearest(array, value:int):
        '''
        Find nearest peak
        by comparing two list
        '''
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    c=0
    d = 0
    peak_found=[]
    for i in peaksref:
        near = find_nearest(peaks,i)
        if near in range(i-size,i+size+1):
            d+=abs(near-i)
            c+=1
            peak_found.append(near)
    return len(list(set(peak_found))),list(set(peak_found)),d

def roi_process(x,drugname,size=100,cal='xcal',flourense=False):
    pickup = 200
    xcopy = np.zeros_like(x)
    if flourense: 
        xt = preprocess(x,smoothing=True,idx=(400,1700),norm=True)
        xc = np.zeros_like(xcopy)
        xc[xt.index[0]-200:xt.index[-1]+1-200] = xt.values.copy()
        x = xc 
    else: 
        x = preprocess(x)
    peak = get_ref_peaks(x,drugname)
    for p,ref in peak.items():
        if ref!=0:
            startref,endref = (ref-size-pickup,ref+(size+1)-pickup)
            start,end = (p-size-pickup),p+(size+1)-pickup
            xcopy[start:end]=x[startref:endref].copy()
    return xcopy
    
    

def roi_process_2(x,drugname,cal='xcal',size=100,flourense=False):
    pickup  = 200
    xcopy = np.zeros_like(x)
    if flourense: 
        xt = preprocess(x,smoothing=True,idx=(400,1700),norm=True)
        xc = np.zeros_like(xcopy)
        xc[xt.index[0]-200:xt.index[-1]+1-200] = xt.values.copy()
        x = xc
    else: 
        x = preprocess(x)

    peaks, properties = find_peaks(x, prominence=prominence)
    peaks_df = pd.DataFrame({"peaks":peaks+pickup,"left":properties['left_bases']+pickup,'right':properties['right_bases']+pickup})
    #peaks_df = peaks_df[(peaks_df['peaks']>400) & (peaks_df['peaks']<1700)]
    #print(peaks_df)
    x = pd.Series(x,index=np.arange(200,2601))
    peak = get_ref_peaks(x,drugname)
    for p,ref in peak.items():
        if ref!=0:
            try:
                l,r = peaks_df[peaks_df['peaks']==ref].values[0].tolist()[1:]
                startref,endref =l-pickup,r-pickup #(ref-size-pickup ,ref+(size+1)-pickup)
                start,end = l-pickup,r-pickup
                xcopy[start:end]=x[startref:endref].copy()
            except: pass
    return xcopy

        
def top_intensity_peaks(data,top=8):
    pdata = preprocess(data).loc[350: 1700]
    peaks = find_peaks(pdata,prominence=0.02)[0]+350
    return pdata.loc[peaks].sort_values(ascending=False).head(top).index.tolist()

############### global variable after process #########################

peaks_df = extract_peaks_from_database(database=database_drugs) ## Loads peaks with names as key and list as value in dict

################################### OLD #########################################

class Alert:
    '''
    Gives the similarity using golden sign and a drug and returns confidence value
    '''
    def __init__(self):
        pass   
       
    def test(self,data,mean_data,threshold=None,show_data=False):
        '''
        Input : data -> drug , mean_data = Golden_scan
        output : return value from 0 - 1
        '''
        p =np.round(cosine_similarity(data.reshape(1,-1),mean_data.reshape(1,-1))[0],5)
        if threshold==None:
            return p
        if show_data==True:
            print(p)
        if p>=threshold:
            return 1
        else:
            return -1

class Drug_Detect_Report():
    def __init__(self,database,drug,solvents=['Water'],scan_level='xcal',confidence=None,peak_modify=None,distance=None,check_with_db=False):
            """
            Drug detect report is a module help to check peaks and unknown peaks from Drug 
            Input : db : database
                drug : Drug Name 
                scan_level : leven of scan which works on x_cal and y_cal
                Optional Input :
                peak_modify : peak_modify helps to give threshold of height og peaks after baseline subtraction
                distance : distance checks the peak for every 50 wavenumber if the max peak value in distance 
                 is greater than threshold
                Confidence : used to check similiarity check between  golden sign and input drug in 
                 Drug_Detect_Report().drug_confidence
            """

            self.drug_list = database.keys()
            self.database = database
            self.check_with_db = check_with_db
            
            solvents = [str(i).capitalize()  for i in solvents]
            if 'Water' not in solvents: solvents.insert(0,'Water')
            drug_formed = drug+'_'+'_'.join(solvents)
            print(f"Looking for drug  = {drug_formed} in Database for instance")
            try:
                self.db=database[drug_formed]
            except:
                try: self.db=database[str(drug)+'_Water']
                except: self.db =  None
                print("DRUG NOT IN DB")
            self.drug=drug
            self.confidence = confidence
            self.peak_modify = peak_modify
            if distance is None : self.distance= 50
            else: self.distance= distance
            self.gold_sign = []
            self.scan_level=scan_level
            self.start = 0
            self.mul=1
            if self.scan_level=='xcal' : self.mul=1
            if self.scan_level=='ycal' : self.mul=1
            self.search_list = self.search_drug_list()

            try:
                if self.scan_level == 'xcal':
                    self.gold_sign = self.pre_process(pd.Series(self.db['raw'],index=np.arange(200,2601)))
                if self.scan_level == 'ycal':
                    self.gold_sign = self.pre_process(pd.Series(self.db['raw'],index=np.arange(350,2152)))
                if self.confidence is None : self.confidence = self.db['confidence']
            except:
                self.gold_sign=[]
                self.confidence = 0
            if self.peak_modify is None : 
                try:
                    self.peak_modify = self.db['peak_threshold']
                except:
                    self.peak_modify = 0.1
            if self.distance is None : 
                try:
                     self.distance =self.db['distance']
                except:
                     self.distance = 20
               
            self.processed=None
       

    def search_drug_list(self):
        l = []
        for i in self.db['peaks'].keys():
            try: l.append(i.split('_')[0])
            except: l.append(i)
        return list(set(l))
    
    def peak_normalization(self,df):
        if len(df)>0:
            if len(df)==2401:
                x = df / (df[100:200].max() - df.min())
                return x
            if len(df)==1802:
                x = df/(df[0:1500].max() - df.min())
                return x
        
    def find_nearest(self,array, value):
        '''
        Find nearest peak
        by comparing two list
        '''
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    
    def findPeaklocation(self,x,name,peak1,peak2):
        '''
        Input : processed scan with peak location and slide range with 10
        Output : Tuple with name and peak location
        '''
        data = 0
        dr = x.loc[peak1:peak2]
        peak = find_peaks(dr,distance=5)[0]
        if len(peak)>0:
            peakd= peak[0]
            data=(name,peakd+peak1)
            #print('Inner ; ',name,peak1,peak2,peak,peakd,peakd+peak1)
        return data
    
    def pre_process(self,x):
        '''
        Performs baseline subtraction and smoothing effect 
        Output : returns : Series 
        '''
        x = butterworth_filter(x)
        smoothness_param,max_iters,conv_thresh=5,10,0.001
        x= x-als_baseline(x,smoothness_param=smoothness_param,max_iters=max_iters,conv_thresh=conv_thresh)
        x = np.where(x>0,x,0)
  
        if  len(x)==1802: x= pd.Series(x,np.arange(350,1802+350))
        if  len(x)==2401: x= pd.Series(x,np.arange(200,2601))
        else: x = pd.Series(x)
        x = self.peak_normalization(x)
        return x   
        
    
    def findings(self,k):
        l=['Water' if str(i).startswith('Water')  else str(i).split('_')[0]  for i,j in k ]         
        return list(set(l))
    
    def detect_of_asset(self,x):
        '''
        Input : Drug 
        Output : returns peaks with name and peak location of peaks from database
        '''
        if self.scan_level=='xcal': self.start = 200
        if self.scan_level == 'ycal': self.start = 350
        x= self.pre_process(x)
        self.processed = x.copy()
        peakoforg=find_peaks(x.loc[:1700],height=self.peak_modify*self.mul,distance=self.distance) # looks peaks only till  water peaks
        peakoforg=peakoforg[0]+self.start
        nn=[self.findPeaklocation(x,i,j[0],j[1])for i,j in self.db['peaks'].items()]
        finding_asset= [i for i in nn if i != 0] # checck peaks for given threshold and extract peak names from databse
        return finding_asset,peakoforg
    

    
    def drug_confidence(self,x,start=450,end=1700,wsize=10,show_data=False,confidence_of_peaks=False):
        '''
        Get Confidence of drug with given drug sample with database
        
        
        Input : Drug : with Ycal data from 350 - 2151
        output :  Peaks : Peaks found in drug, 
        found : finding_asset is the peaks with labeled data is given like water peak in 1650 eg out : (lido,1275),
        Known : refers to peaks found with peak location of +15 or -15, 
        Unknown :refers to peaks which are not know.
        Flag_Unknown_peaks : Is a flag which will be 1 when length of Unknown Peaks is more than 2 and Optional 
         if confidence of spectrum till weater peak is mmore than Threshold given to module
        Optional Output : if Golden _ scan : returns Confidence of spectrum with golden scan
        
        '''
        limit= 1600
        d={}
        d['peaks']=[]
        d['found_drug']=[]
        d['found_drug_peaks']=[]
        d['known_peaks']=[]
        d['unknown']=[]
        d['flag_of_unknown_peaks']=1
        #d['confidence_full_spectrum']=0
        #d['score']=99999999
        peaks=[]
        xcopied = np.array(x).copy()
        if self.db is None:
            x = self.pre_process(x)
            peaks=list(find_peaks(x.iloc[:limit].values,distance=self.distance,threshold=self.peak_modify)[0])
            p=0
            
        flag_unknown_peaks= 0
        try:
            finding_asset,peaks= self.detect_of_asset(x)
            finding_asset=sorted(finding_asset,key=lambda x: x[1])
            if len(self.gold_sign)>0:
                alert=Alert()
                p=alert.test(self.processed.loc[start:end].values,self.gold_sign.loc[start:end].values)[0] # confidence check
            else: p  = 0
        except:
            finding_asset=[]
            p=0
     
        if len(finding_asset)>0 :
            found = list(set(list(zip(*finding_asset))[0])) # list the peaks with names in list

            unknown=[i for i in peaks if i<limit and i not in list(list(zip(*finding_asset))[1])] # finds unknow peaks when golden scan not present
            known,golden_peak=list(list(zip(*finding_asset))[1]),0
            if len(self.gold_sign)>0:
                golden_peak = find_peaks(self.gold_sign.loc[:limit],height=self.peak_modify,distance=self.distance)[0]+self.start
                known = [i for i in unknown if abs(self.find_nearest(golden_peak, i )-i)<=wsize]
                unknown = [i for i in unknown if abs(self.find_nearest(golden_peak, i )-i)>wsize] # derving unknow by window size 15
                unknown= [i for i in unknown if (i>=start and i<=end)]
                promising_peaks = list(self.processed[find_peaks(self.pre_process(xcopied),prominence=0.5)[0]+200].index) # checking store peak
                unknown=[peaks_each for peaks_each in promising_peaks if peaks_each in unknown] # cehcking store peak in unknown
                if len(unknown)>2 and  p<self.confidence: flag_unknown_peaks = 1;  # Flag logic 
            elif len(unknown)>2: flag_unknown_peaks=1       
            #Creation of report
            d['peaks'] = list(peaks)
            d['found_drug'] = sorted(self.findings(finding_asset)) #if flag_unknown_peaks==0 else ['Water'] if 'Water' in sorted(self.findings(finding_asset)) else []
            d['not_found_drug'] = [dk for dk in self.search_list if dk not in d['found_drug']]
            d['found_drug_peaks']=finding_asset #if flag_unknown_peaks==0 else []
            d['known_peaks']=known
            d['unknown'] = unknown
            d['flag_of_unknown_peaks'] = flag_unknown_peaks
            #if p!=0: d['confidence_full_spectrum']=p
            if show_data==True:
                print("Peak_modify = {}, Confidence = {}, distance = {}".format(self.peak_modify*self.mul,self.confidence,self.distance))
                print("Gold sign peak = {}".format(golden_peak))
                print("Peak = {}\nFound = {}".format(peaks,finding_asset))
                print("Known Peaks (unlabeled) = {}".format(known))
                print("Unknown Peaks = {}".format(unknown ))
                print("Confidence_full_spectrum = {}".format(p))
                print("Flag of Unknown peaks = {}".format(flag_unknown_peaks))
                
            if confidence_of_peaks==True and len(self.gold_sign)>0:
                alert = Alert()
                score = 0
                for i in self.db['peaks'].keys():
                    
                    if i !='Water':
                        range_value = self.db['peaks'][i]
                        check = manhattan_distances([self.processed.loc[range_value[0]:range_value[1]].values],[self.gold_sign.loc[range_value[0]:range_value[1]].values])
                        check1 = euclidean_distances([self.processed.loc[range_value[0]:range_value[1]].values],[self.gold_sign.loc[range_value[0]:range_value[1]].values])
                        value = cosine_similarity([self.processed.loc[range_value[0]:range_value[1]].values],[self.gold_sign.loc[range_value[0]:range_value[1]].values])
                        #value = self.custom_eucl(self.processed.loc[range_value[0]:range_value[1]].values,self.gold_sign.loc[range_value[0]:range_value[1]].values)
            
                        if show_data==True:
                            print("{}  = {}".format(i,value))
                        try:
                            if np.isnan(value[0][0]):
                                    value=0
                                    score+=(1-0)
                            else:
                                score+=(1-value[0][0])
                        except: pass


                #d['score']=1-(score/len(self.db['peaks'].keys()))
                #print(d['score'])
            return d
        else:
            d['peaks']=peaks
            d['unknown']=peaks
            return d
      
    
    def custom_eucl(self,x,y):
        '''1/(1+sqrt(sum((n1 - n2) .^ 2)))'''
        return 1/(1+np.sqrt(sum((x-y)*2)))
        
    def drug_manhattan_distance(self,start=400,end=1400):
        if((self.processed is not None) and len(self.gold_sign)>0):
            return manhattan_distances([self.processed.loc[start:end].values],[self.gold_sign.loc[start:end].values])[0][0]
    def drug_euclidean_distance(self,start=400,end=1400):
        if((self.processed is not None) and len(self.gold_sign)>0):
            return euclidean_distances([self.processed.loc[start:end].values],[self.gold_sign.loc[start:end].values])[0][0]
       
        
class Absolute_ls_score:
    def __doc__(self):
        return '''
                Input - > x : Input , y = Raw _database_spectrum, start = Windows_start_size, end  = Windows_end_size.
                output : Dict -> LS, Corr
                '''
    
    def __init__(self,x,y,start=300,end=1500):
        self.x = x 
        self.y = y
        self.start = start #exp_2
        self.end = end
        self.components = np.array([self.preprocess_ls(y)])
        self.corr = 0
        self.ls = 0
        self.ssim = 0 
        
        
    def get_score(self,plot=False):
        query_spectrum =  self.subtract_data_ls(self.x,self.y,plot=plot)
        self.ls =  1-linear_model.LinearRegression().fit(self.components.T, query_spectrum).coef_[0]    
        return {"ls":self.ls,'corr':self.corr,'ssim':self.ssim}


    def preprocess_ls(self,x,norm=True,null=True):
        """
        Does baseline correction and nullify the data upon peak_normalization is preformed

        """
        def peak_normalization(df):
            if len(df)>0:
                #print(type(df))
                x= df[self.start:self.end]
                x = x / (x.max() - x.min())
                return x

        x = butterworth_filter(x)
        #x -= base.polyfit_baseline(x,deg=7)
 
        x -= airpls_baseline(x)
        #x -= base.fabc_baseline(x)    #exp_1
        #x -= base.als_baseline(x)
        #x =  convex_baseline_adjustment(x,itr=350)
        if null: x= np.where(x>0,x,0)

        if norm:x = peak_normalization(x)
        df = pd.Series(x,index=np.arange(len(x)))
        if len(x)==1802: df = pd.Series(x,index= np.arange(350,2152))
        if len(x)==2401: df = pd.Series(x,index = np.arange(200,2601))
        #print(f'Shape = {df.shape}')
        return df

    def subtract_data_ls(self,x,y,method='preprocess',plot=False):
        """if method=='second':
            x = second_der_v2(butterworth_filter(x))
            y = second_der_v2(butterworth_filter(y))
            data = x - y
            data=data.apply(lambda x: np.where(x>0,x,0))"""
        if method=='preprocess':
            x =  self.preprocess_ls(x)
            y = self.preprocess_ls(y) 
            ar = Alert()
            self.ssim = ar.test(x.values,y.values)[0]
            data = abs(x - y)
            corr,_ = pearsonr(x,y)
            self.corr = corr
            data = data.apply(lambda x: np.where(x>0,x,0))
        return data



def check_critical_peak(db,drugname,peaks,size=7,assigned_peaks=[]):
    '''
    database,drugname,peaks,size
    return bool, true, false
    
    if no peaks in db then c ==  0, ie True
    '''
    def find_nearest(array, value):
        '''
        Find nearest peak
        by comparing two list
        '''
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    try:
        c_peaks = db[drugname]['critical_peak']
    except: 
        if len(assigned_peaks)>0: c_peaks=assigned_peaks
        else: return None
    c = 0
    nears = []
    if len(c_peaks)>0 and len(peaks)>0:
        for i in c_peaks:
                near = find_nearest(peaks,i)
                if near not in nears:
                    if near in range(i-size,i+(size+1)): #condition change
                        c+=1
                        nears.append(near)
    #print(c_peaks)
    if c==len(c_peaks): return True  
    else: return False


    
def peak_normalization(df):
        if len(df)>0:
            df = np.array(df)
            if len(df)==2401:
                x = df / (df.max() - df.min())
                return x
            if len(df)==1802:
                x = df/(df[0:1500].max() - df.min())
                return x
 

def drug_pass_fail(x,db,drugname):
            c = 0 
            report = Drug_Detect_Report(database=db,drug=drugname)
            drugreport = report.drug_confidence(x,confidence_of_peaks=True,start=350,end=1650)
            if len(report.processed)==2401: pickup =200
            if len(report.processed)==1802: pickup = 350
            def preprocess_ls(x,norm=True,null=True):
                """
                Does baseline correction and nullify the data upon peak_normalization is preformed

                """
                def peak_normalization(df):
                    if len(df)>0:
                        x = df
                        x = x / (x.max() - x.min())
                        return x

                #x = butterworth_filter(x) exp_6
                #x =  convex_baseline_adjustment(x,itr=200) # exp_5
                smoothness_param,max_iters,conv_thresh=5,10,0.001 #params for testsing
                x-=airpls_baseline(x,smoothness_param=smoothness_param,max_iters=max_iters,conv_thresh=conv_thresh)
                if null: x= np.where(x>0,x,0)

                if norm:x = peak_normalization(x)
                df = pd.Series(x,index=np.arange(len(x)))
                if len(x)==1802: df = pd.Series(x,index= np.arange(350,2152))
                if len(x)==2401: df = pd.Series(x,index = np.arange(200,2601))
                return df
            
            #preprocess incoming scan
            x = np.array(x)
            raw_norm = preprocess_ls(x)
     
            #finding peaks 
            peaks =  find_peaks(raw_norm,db[drugname]['peak_threshold'])[0] + pickup #exp_4
            peak_score=drugreport['score'] if drugreport['score']!=99999999 else 0 #peaks score from Drug Report 
            peak_found,got_peaks,_=number_of_peaks(peaks,db[drugname]['peaks_list'])
  
             # check for critical_peak returns turn or None
            try:
                critical_peak = db[drugname]['critical_peak']   # check critical_peak in db 
            except:
                critical_peak = []
    
            c = check_critical_peak(db=db,drugname=drugname,peaks=peaks,size=10,assigned_peaks=critical_peak)
           
            if c==True or c is None:     
                y = db[drugname]['raw']

                if len(y)>0:
                    ab =  Absolute_ls_score(x,y)
                    data = ab.get_score()
                    ls = data['ls']
                    corr = data['corr']
                    ssim = data['ssim']
                    peaks_found_out = peak_found/len(db[drugname]['peaks']) 
                    peak_score = peak_score
                    # scoring if structure present 
                    score = (ssim * 0.2) + (ls * 0.2) + (corr * 0.2) + (peaks_found_out * 0.2) + (peak_score * 0.2)
                    
                    
                    # check for peak Threshold 
                    try:
                        pass_threshold = db[drugname]['pass_threshold']  # pass_threshold for drug name if present
                    except:
                        pass_threshold = 0.75 # default pass_threshold
                        
                    
                    #pass fail check
                    if  drugname in drugreport['found_drug'] and score>pass_threshold:    #score reduced for 55 for R178
                        return drugname, round(round(score,3)*100,2),got_peaks
                    else: return 'None',round(round(score,3)*100,2),got_peaks # fail return score 
                    
                else:
                    
                    # if structure is not present only with peaks found scoring 
                    score = peak_found/len(db[drugname]['peaks'])
                    if score >= 0.7: # deault peak found thresold
                        return drugname ,round(round(score,3)*100,2),got_peaks
                    else: 
                        return "None",round(round(score,3)*100,2),got_peaks
            else:
                return 'None',0,[]

            
            
def get_pass_fail_old(spectra, selected_name):
    result = 'fail'
    try:
        found_name, score,peaks_found = drug_pass_fail(x=spectra, db=database_drugs, drugname=selected_name)
        peaks_found = np.array(peaks_found).tolist()
        peaks_range = get_range_of_peaks(found_name)
        golden_peaks= database_drugs[found_name]['peaks_list']
    except Exception as e:
        print("Failed scan test ",e)
        found_name = 'Unknown'
        score = 0
        peaks_found = []
        if len(spectra)==2401:peaks_range = (200,2601)
        if len(spectra)==1802: peaks_range =(350,2152)
        golden_peaks=[]
    if found_name == selected_name:
        result = 'pass' 

    try:
        golden_signature = database_drugs[selected_name]['gold_sign']
    except:
        print("No Golden Signature")
        golden_signature = []

    
        
    
    data_to_return = {
        "found":found_name,
        "gold_sign":golden_signature,
        "result": result,
        "score":score,
        "peaks_found":peaks_found,
        "peaks_range":peaks_range,
        "golden_peaks": golden_peaks
    }
    return data_to_return
   

def drug_identification(x):
    l=[]
    found_name = 'None'
    score_out = 0
    
    for i in IDENTIFY_DRUG:
        if i in database_drugs.keys():
            d={}
            d['Drug']=i
            try:
                data = get_pass_fail_old(x,i)
                result = data['result']
                score = data['score']
                peaks_found = data['peaks_found']
            except Exception as e:
                print(f"Failed scan test for drug - {i} Error - {e} ")
                result,score,peaks_found='fail',0,[]
            d['status']=result
            d['score']=score
            d['peaks']=peaks_found
            l.append(d)
    df = pd.DataFrame(l) 
    try:
        drug_one = df[df['status']!='fail'].sort_values('score',ascending=False).iloc[0]
        found_name = drug_one['Drug']
        score_out = drug_one['score']
        peaks_out = drug_one['peaks']
        del df,l
        return found_name,score_out,peaks_out
    except Exception as e: 
        print(f'Score cant find because  of = {e}')
        return "None",0,[]
    

def identify_drug_old(spectra):
    
    try:
        found_name, score,peaks_found = drug_identification(x=spectra)
        if found_name == 'None' or score == 0: found_name = 'Unknown'
        if found_name != 'Unknown': 
            peaks_range = get_range_of_peaks(found_name)
            golden_peaks= database_drugs[found_name]['peaks_list']
        else:
            if len(spectra)==2401:peaks_range = (200,2601)
            if len(spectra)==1802: peaks_range =(350,2152)
            golden_peaks= []
            

    except Exception as e:
        print("Failed Identification test ",e)
        found_name = 'Unknown'
        score = 0
        peaks_found=[]
        if len(spectra)==2401:peaks_range = (200,2601)
        if len(spectra)==1802: peaks_range =(350,2152)
        golden_peaks=[]
        
    
    try:
        golden_signature = database_drugs[found_name]['gold_sign']
    except:
        print("No Golden Signature")
        golden_signature = []


    # no result 
    data_to_return = {
        "found":found_name,
        "gold_sign":golden_signature,
        "score":score,
        "peaks_found":peaks_found,
        "peaks_range":peaks_range,
        "golden_peaks": golden_peaks
    }
    
    return data_to_return

#################################################### NEW ###############################################
################### Preprocess ############################
def preprocess(x,norm=True,null=True,smoothing=False,idx=None,ext=False):
                global size
                """
                Does baseline correction and nullify the data upon peak_normalization is preformed

                """
                x = np.array(x)
                if idx is not None: 
                    u_index = np.arange(idx[0],idx[1])
                    if len(x)==2401: x = x[idx[0]-200:idx[1]-200]
                    if len(x)==1802: x = x[idx[0]-350:idx[1]-350]
                        
                def peak_normalization(df):
                    if len(df)>0:
                        x = df
                        x = x / (x.max() - x.min())
                        return x
                if smoothing: x = butterworth_filter(x) #exp_6
                #x =  convex_baseline_adjustment(x,itr=200) # exp_5
                smoothness_param,max_iters,conv_thresh=5,10,0.001 #params for testsing
                if ext==True: smoothness_param,max_iters,conv_thresh=2,20,0.0001
                x-=airpls_baseline(x,smoothness_param=smoothness_param,max_iters=max_iters,conv_thresh=conv_thresh)
                if null: x= np.where(x>0,x,0)

                if norm:x = peak_normalization(x)
                df = pd.Series(x,index=np.arange(len(x)))
                if idx is not None: df = pd.Series(x,index=np.arange(idx[0],idx[1]))
                if len(x)==1802: df = pd.Series(x,index= np.arange(350,2152))
                if len(x)==2401: df = pd.Series(x,index = np.arange(200,2601))
                return df

def normalize_to_peak(df_list, peak=0):

    if len(df_list)==2401: df= pd.DataFrame(df_list, index= np.arange(200,2601)).T
    elif len(df_list)== 1802: df= pd.DataFrame(df_list, index= np.arange(350,2152)).T
    peak= peak[0]

    try:
        
        if not 340<=peak<=2160: peak=0
        window = 10
        
        if (df.max(axis=1) - df.min(axis=1))[0] == 0 or (df.loc[:,peak-window:peak+window].max(axis=1) - df.min(axis=1))[0] == 0:
            return df
        
        if peak == 0:
            df = df / (df.max(axis=1) - df.min(axis=1))[0]
        else : 
            df = df / (df.loc[:,peak-window:peak+window].max(axis=1) - df.min(axis=1))[0]
        return df

    except Exception as e:
        print(e)
        return pd.DataFrame()



####################### Info Extractor #############################   
def info_data(x,drug,flourense=False):
    global prominence
    global database_drugs
    if isinstance(x,pd.DataFrame):
        l=[]
        for i in range(x.shape[0]):
            d={}
            y =  x.iloc[i]
            raw_norm = preprocess(y)
            pickup = 200 if len(raw_norm)==2401 else 350
            peaks =  find_peaks(raw_norm,prominence=prominence)[0] + pickup
            tol_peaks=database_drugs[drug]['peaks_list']
            num_of_peaks,peaks_got,diff = number_of_peaks(peaks,tol_peaks)
            d['id']=i
            d['Num']=num_of_peaks
            d['peaks']=peaks_got
            d['diff']=diff
            d['percentage'] = round((num_of_peaks/len(tol_peaks))*100,2)
            del x,raw_norm,xcopy
            l.append(d) 
        return pd.DataFrame(l)
    if isinstance(x,pd.Series) or isinstance(x,list) or isinstance(x,np.ndarray):
            xcopy = x.copy()
            pickup = 200 if len(xcopy)==2401 else 350
            if flourense: 
                pickup=400
                x = pd.Series(x,index=np.arange(200,2601))
                raw_norm = preprocess(x,smoothing=True,idx=(400,1700))
            else:
                raw_norm = preprocess(x)
            peaks =  find_peaks(raw_norm,prominence=prominence)[0] + pickup
            """dummy = pd.Series(xcopy,index=np.arange(200,2601))   
            peaks_values = dummy[peaks]   
            plt.scatter(peaks_values.index,peaks_values.values)
            plt.plot(dummy)"""
            tol_peaks=database_drugs[drug]['peaks_list']
            num_of_peaks,peaks_got,diff = number_of_peaks(peaks,tol_peaks)
            percentage = round((num_of_peaks/len(tol_peaks))*100,2)
            del x,raw_norm,xcopy
            return num_of_peaks,peaks_got,diff,percentage

def structure_score(x,drugname,size=100,cal='xcal',flourense=False):
    #print(f"test on  = {drugname}")
    global database_drugs
    peaks = database_drugs[drugname]['peaks_list']   
    ref = database_drugs[drugname]['raw']
    got_data = info_data(x,drugname,flourense=flourense)

    if flourense:processed_scan = list(preprocess(x,smoothing=True).values)
    else: processed_scan = list(preprocess(x).values)

    if len(ref)>0:
        if drugname == "Lidocaine": rop =roi_process_2
        else: rop = roi_process
        test_roi  = rop(x,drugname,cal=cal,size=size,flourense=flourense)
        ref_roi = rop(ref,drugname,cal=cal,size=size,flourense=flourense)
        
        test = test_roi
        ref = ref_roi
        
        pvalue = Alert()
        ssim = round(pvalue.test(test,ref)[0]*100,3)
        corr = round(pearsonr(test, ref)[0] * 100, 3) if not math.isnan(pearsonr(test, ref)[0]) else 0
        peaks_found  = got_data[-1]
        percent = round(sum([ssim,corr,peaks_found])/3,3)
        return {"ssim":ssim,"corr":corr,"peaks":peaks_found,"percent":percent,"y_processed":processed_scan}
    else: 
        return 0
    

def compute_score(test,drugname=None,check_intensity=True,flourense=False): 
    global database_drugs
    global prominence
    w_size=50  #experiment_1
    copydatat = pd.Series(np.zeros_like(test),index=np.arange(200,2601))
    copydatar = pd.Series(np.zeros_like(test),index=np.arange(200,2601))
    pickup = 200 if len(test)==2401 else 350
    if flourense: pickup = 200
    test = pd.Series(test,index=np.arange(200,2601))  
    
    g_peaks = database_drugs[drugname]['peaks_list']
    
    if flourense:
        ref = pd.Series(database_drugs[drugname]['raw'],index=np.arange(200,2601))
        ref = preprocess(ref,smoothing=True,idx=(400,1700))
        copydatar.loc[ref.index[0]:ref.index[-1]] = ref.values.copy()
        ref = copydatar.copy()
        testp =  preprocess(test,smoothing=True,idx=(400,1700))
        copydatat.loc[testp.index[0]:testp.index[-1]]= testp.values.copy()
        testp = copydatat.copy()

        ref = ref.values
        testp =  testp.values

    else:
        ref = preprocess(database_drugs[drugname]['raw']).values
        testp =  preprocess(test).values
    
    processed_scan = list(preprocess(test).values)
   
    
    
    ref_peaks = find_all_near_peaks(find_peaks(ref,prominence=prominence)[0]+pickup,g_peaks)
    test_peaks = find_all_near_peaks(find_peaks(testp,prominence=prominence)[0]+pickup,g_peaks,multi=True,strict_mode=True)
    test_peaks_P = find_all_near_peaks(find_peaks(testp,prominence=prominence)[0]+pickup,g_peaks)
    
    # peaks_found = [i for i,j in test_peaks_P.items() if j!=0]
    ### ref peak shown if it is within 2 wavenumber else actual peak is displayed.
    shift_window=2
    peaks_found = [j if abs(i-j)> shift_window else i for i,j in test_peaks_P.items() if j!=0]
    
    peaks_found_score = []
    peak_con ={}  
    for i,j in ref_peaks.items():
        ref_part = ref[j-pickup-w_size:j-pickup+w_size]
        maxd=0
        maxp = 0
        if len(test_peaks[i])>0:
            for k in test_peaks[i]:
                test_part = testp[k-pickup-w_size:k-pickup+w_size]
                ins = abs(max(test_part)-max(ref_part))
                pvalue = Alert()
                score = round(pvalue.test(test_part,ref_part)[0]*100,3)
                score = score * dynamic_score(int(abs(k-i)))
                
                #experiment_3 to check on corr in place of ssim
                #score = round(pearsonr(test_part,ref_part)[0]*100,3)
                
                if maxd<score: 
                    maxd = score
                    maxp = k
            if check_intensity:
                if maxp!=0:
                    test_part_peak_int = testp[maxp-pickup]
                    ref_part_peaks_int = ref[j-pickup]
                    diff_intensity = round(abs(test_part_peak_int - ref_part_peaks_int),2)
                    score_intensity = dynamic_score(diff_intensity,mode='intensity')*100
                    peak_con[i]=np.mean((maxd,score_intensity))
                else: peak_con[i]=0
            else:
                peak_con[i]=maxd
        if maxp!=0: 
            peaks_found_score.append(maxp)
    
    sum_of_con = round(sum([val for key,val in peak_con.items()])/len(g_peaks),3)
    return {"score":sum_of_con,"peaks_found":peaks_found,"y_processed":processed_scan,"ref_peaks":ref_peaks,"test_peaks":test_peaks,"frame":peak_con}

        
def peak_location_score(test,drugname=None,frame=False): 
    global database_drugs
    global prominence
    score = {}
    ss = 0
    score_return = 0
    pickup = 200 if len(test)==2401 else 350
    test = pd.Series(test,index=np.arange(200,2601))  
    def dynamic_score_for_peaks(p,mode="shift"):
        def quar_func(k, a, b, c):
                    return (a*(k**2)) + (k*b) +c
        if mode=='shift':
            if isinstance(p,int):
                x = [0,1,2,3,4,5,6,7,8,9]
                y = [1,0.99,0.98,0.97,0.96,0.95,0.93,0.91,0.88,0.84]
                popt, pcov = curve_fit(quar_func, x, y)
                x_test = np.arange(0,p+2)
                data = quar_func(x_test,*popt)
                return round(data[list(x_test).index(p)],2)
    g_peaks = database_drugs[drugname]['peaks_list']
    
    check_fl = flourencense_identify(test,just_peaks=True)
    if check_fl:
        ref = pd.Series(database_drugs[drugname]['raw'],index=np.arange(200,2601))
        ref = preprocess(ref,smoothing=True,idx=(400,1700)).values
        test1 = preprocess(test,smoothing=True,idx=(400,1700)).values
        pickup = 400
    else:
        ref = preprocess(database_drugs[drugname]['raw']).values
        test1 =  preprocess(test).values
      
    processed_scan = list(test1)
    
    ref_peaks = find_all_near_peaks(find_peaks(ref,prominence=prominence)[0]+pickup,g_peaks)
    test_peaks = find_all_near_peaks(find_peaks(test1,prominence=prominence)[0]+pickup,g_peaks,multi=True,strict_mode=True)

    for i,j in test_peaks.items():
        min_diff = 999
        if drugname in NARCO_DRUG and i in database_drugs[drugname]['critical_peak'] and len(j)==0:   
            score_return-=25
            
        for k in j:
            if abs(i-k)<min_diff:
                min_diff = abs(i-k)
        if len(j)>0: dscore = dynamic_score_for_peaks(int(min_diff),mode='shift')
        else: dscore  = 0
        score[i] = {"score":dscore,"found":j}
        ss+=dscore
    
    score_return += round(((ss/len(g_peaks))*100),3)   
    #return {"score":sum_of_con,"peaks_found":peaks_found,"y_processed":processed_scan}
    if frame: 
        return test_peaks,score,score_return
    
    return score_return

########################### Pass Fail ######################################

def get_pass_fail(spectra, selected_name,flourense=False):
    result = 'fail'
    reduce_score = False
    if not (isinstance(spectra,np.ndarray) or isinstance(spectra,pd.Series)):
            spectra = np.array(spectra)

    output = structure_score(spectra,selected_name,flourense=flourense)
    ssim,corr,peaks_percent_in_range,percent = output['ssim'],output['corr'],output['peaks'],output['percent']
    output = compute_score(spectra,selected_name,flourense=flourense)
    # y_processed = output['y_processed']
    if flourense:
        y_processed = preprocess(spectra,smoothing=True,idx=(400,1700))
        peaks_in_spectra = find_peaks(y_processed,prominence=prominence)[0]+400

    else:
        y_processed = preprocess(spectra)
        peaks_in_spectra = find_peaks(y_processed,prominence=prominence)[0]+200
    peaks_found = np.array(output['peaks_found']).tolist()
    found_name =  selected_name
    peaks_range = get_range_of_peaks(selected_name)
    golden_peaks= database_drugs[selected_name]['peaks_list']
    
    #score = round(((ssim+corr+peaks_percent_in_range+output['score'])/4 ),3)  ##### Old Matrix
    score  =  round((percent+output['score'])/2,3)  ### New Matrix

    #print(f"""Drug = > {selected_name}, SSIM = {ssim},Corr = {corr}, Peaks inrange = {peaks_percent_in_range},Compute = {output['score']}, total score = {score}""")
    #if (selected_name in NARCO_DRUG) or True:
    
    # peaks_in_spectra = find_peaks(y_processed,prominence=prominence)[0]+200
    if len(y_processed)==2401: pickup = 200
    if len(y_processed)==1802: pickup = 350
    if 'critical_peak' in database_drugs[selected_name].keys() and len(database_drugs[selected_name]['critical_peak'])>0:
                if check_critical_peak(database_drugs,selected_name,peaks_in_spectra)==False:
                    print(selected_name,'reduced')
                    reduce_score = True
                    
      
    if reduce_score:
        score-=25
        if score<0: score = 0
        
    if score>70:
        result = 'pass' 

    try:
        # golden_signature = database_drugs[selected_name]['gold_sign']
        golden_signature = list(preprocess(database_drugs[selected_name]['raw']).values)   

    except:
        print("No Golden Signature")
        golden_signature = []
        


    data_to_return = {
        "found":found_name,
        "gold_sign":golden_signature,
        "result": result,
        "score": score,
        "peaks_found":peaks_found,
        "peaks_range": peaks_range,
        "golden_peaks": golden_peaks,
        "y_processed": output['y_processed'],
        
    }
    
    return data_to_return


##############################Extention###############################

def ext_method(inp,d_name):
    def norm(x):
        x = np.array(x)
        x =  (x-x.min())/(x.max()-x.min())
        return x
    
    def second_order(x):
        x = butterworth_filter(x)
        dp = np.gradient(x)
        for _ in range(1):
            dp = np.gradient(dp)
        dp = norm(dp)
        return dp
    
    def to_series(x):
        if len(x)==1802: return pd.Series(x,index=np.arange(350,2152))
        elif len(x)== 2601: return pd.Series(x,index=np.arange(200,2601))
        else: return pd.Series(x)
    
    data_name = []
    for i in range(1,len(d_name)):
        for drug in list(combinations(d_name,i)):
            d = {}
            d['drug']=drug
            d['est']=estimate_structure(data=[database_drugs[dr]['raw'] for dr in drug])
            min_x,max_x = 1000,0
            for dr in drug:
                min_x = round(min(database_drugs[dr]['peaks_list']),-2) if min_x>=min(database_drugs[dr]['peaks_list']) else min_x
                max_x = round(max(database_drugs[dr]['peaks_list']),-2) if max_x<=max(database_drugs[dr]['peaks_list']) else max_x
            d['drug_peak']=(min_x,max_x)
            
            pre_d = second_order(to_series(d['est']))#.values
            inp_p_d = second_order(to_series(inp))#.values
            d['corr_d'] =  pearsonr(inp_p_d,pre_d)[0]
            
            pre = preprocess(d['est'],norm=True,smoothing=True,idx=(500,1700),ext=True).values
            inp_p = preprocess(inp,norm=True,smoothing=True,idx=(500,1700),ext=True).values

            sub_d = inp_p - pre
   
            d['sub'] = np.where(sub_d>0,sub_d,0)
            d['sub_p'] = np.where(d['sub']<inp_p,0,d['sub'])
            d['area_under'] = np.trapz(d['sub_p'])
            data_name.append(d)

    if len(data_name)>0:
        edf=pd.DataFrame(data_name)
        edf['area_p']=1-(edf['area_under']/edf['area_under'].max())
        edf['ss']= (edf['area_p']*0.7) + (edf['corr_d']*0.3)
        
        edf=edf.sort_values(by='ss',ascending=0)
        
        return edf
    else: return pd.DataFrame([])

def filter_data_ext(df):
    if len(df)>0:
        top = 3
        
        def diff_score_in(x,y,diff=0.03):
            if (abs(x-y) >= diff): return True
            else: return False
        def most_occurate(x:dict,top=2):
                values = pd.DataFrame([x]).T.reset_index()
                values.rename(columns={"index":'drug',0:'score'},inplace=True)
                values = values.nlargest(top,columns='score')
                values['diff']=values.loc[:,'score'].diff().abs().fillna(0)
                names_k = []
                for i,j in values.iterrows():
                    if j['diff']>2:
                        break
                    names_k.append(j['drug'])
                return follow_order(names_k)

        def follow_order(x):
            l =[]
            for i in x:
                if i not in l: l.append(i)
            return l
                
        if len(df.drug.iloc[0])==1:
    
            if diff_score_in(df.iloc[0]['area_p'],df.iloc[1]['area_p']): 
                    return follow_order([i for i in df.drug.iloc[0]])
            elif len(df.drug.iloc[1])==1:
                return follow_order([j for i in df.drug.iloc[:2] for j in i])
            else:
                names_got= [j for i in df.sort_values(by='ss',ascending=0).drug.head(4).values for j in i ]
                if len(list(set(names_got)))>=3: top=3
                if len(list(set(names_got)))<3: top =2
                return most_occurate(dict(Counter(names_got)),top=top)
        
        elif len(df.drug.iloc[0])==2:

            if diff_score_in(df.iloc[0]['ss'],df.iloc[1]['ss'],diff=0.1): 
                    return follow_order(([i for i in df.drug.iloc[0]]))
            else:
                names_got= [j for i in df.sort_values(by='ss',ascending=0).drug.head(4).values for j in i ]
                if len(list(set(names_got)))>=3: top=3
                if len(list(set(names_got)))<3: top =2
                return most_occurate(dict(Counter(names_got)),top=top)
            
        elif len(df.drug.iloc[0])==3:
            return follow_order([i for i in df.drug.iloc[0]])
            
        else:
            names_got= [j for i in df.sort_values(by='ss',ascending=0).drug.head(4).values for j in i ]
            return most_occurate(dict(Counter(names_got)),top=3)
    else:
        return []
    
def select_drug_rdf(df):
    try:
            #print(len(df[df['score']>30])==0),round(df.iloc[0]['score'])>90 and round(df.iloc[1]['score'])>80)
            if len(df[df['score']>50])==0: top = 0
            elif (round(df.iloc[0]['score'])>=90 and round(df.iloc[1]['score'])>=80): top = 2
            elif round(df.iloc[1]['score']) <=50: top= 1 # if second score is less than 50
            elif round(df.iloc[0]['score'])>95: top = 1
            #elif df.iloc[1]['score']>((70*df.iloc[0]['score'])/100): # if second top is more than 70% of first score
                    #top=2
            elif abs(round(df.iloc[0]['score']) - round(df.iloc[1]['score'])) >=35: top = 1
            elif df.iloc[1]['score'] <df.iloc[0]['score']-0.4*df.iloc[0]['score']: top=1  #### New added if 2nd result is less than 40% of first
            # ###or 
            # elif df.iloc[1]['score'] <df.iloc[0]['score']-0.4*df.iloc[0]['score']:
            #     top=1
            #     for i in range(len(df)-1):
            #         if df.iloc[i+1]['score'] <df.iloc[i]['score']-0.4*df.iloc[i]['score']: break
            #         else: top+=1
            else: top = len(df[df['score']>=50])
    except:top = 2
    if top>2: top = 2
    print('Top = > ',top)
    if top!=0: return df.head(top).drugname.values.tolist()
    else: return None

def expection_checker(spectra,found):
            
    window_size =  4
    
    def norm(x):
        x = np.array(x)
        x =  (x-x.min())/(x.max()-x.min())
        return x
    
    peaks = find_peaks(norm(spectra),prominence=0.1)[0]+200
    peaks = list(filter(lambda x: x>450 and x<1700,peaks))
    
    peaks_of_sample = []
    for k in found: peaks_of_sample.extend(database_drugs[k]['peaks_list'])

    known = list(filter(lambda x: any(map(lambda y: abs(x-y)<=window_size, peaks_of_sample)), peaks))
    unknown =  list(filter(lambda x: x not in known, peaks))
    
    lst =  []
    for i in IDENTIFY_DRUG:
        if i not in found or i not in NARCO_DRUG:
            kl = list(filter(lambda x: any(map(lambda y: abs(x-y)<=window_size,unknown)),database_drugs[i]['peaks_list']))
            d={}
            d['drug']=i
            d['found_peaks'] = kl
            d['peaks_len'] = len(kl)
            lst.append(d)
    df = pd.DataFrame(lst)
    df  = df.sort_values('peaks_len',ascending=False)
    if len(found)==2: add = 1
    if len(found)==1: add = 2
    return df.drug.head(add).values.tolist()

def extreme_mapper(x):
    mapper = {
        ('Caffeine','Phenacetin'): "Lidocaine"
    }
    x = tuple(sorted(list(map(lambda x1: str(x1).capitalize(),x))))
    if x in mapper.keys():return mapper[x]
    else: return None

def syr_score_check(spectra,drugname,flourense=False):
    output = structure_score(spectra,drugname,flourense=flourense)   
    return output['percent']   
######################### Identification ##############################
     
def identify_proba(spectra,flr=False,return_frame=False):
    # from oak_utilities import cancel_event,abort_if_requested
    check_v3 = False
    names,edf=0,0
    default_data = [{
                    'drugname':"Unknown",
                    "score":0,
                    "y":list(spectra),
                    'gs_peaks':[],
                    "range":(200,2601),
                    "peaks_found": top_intensity_peaks(spectra),#[int(i)+200 for i in list(find_peaks(list(preprocess(spectra).values),prominence=0.6)[0])[:7]],
                    "y_gs":[],
                    "y_processed":list(preprocess(spectra).values)      
                }]

    frame  = []
    def update_score(df,drug,score):
        if isinstance(df,pd.DataFrame):
            df.loc[df['drugname']==drug,'score']=score
        return df 
    def peaks_percentage(df):
        pf = len(df['peaks_found'])
        dpf = len(database_drugs[df['drugname']]['peaks_list'])
        if dpf>2:dk = round((pf/dpf) * 100,3)
        else: dk = 40
        return dk
    check_fl1 = flourencense_identify(spectra)
    print(f"Flourense Tested on 1-> {check_fl1}") 
    
    if flr==True: 
        print(f"From Camera Shutter -> {flr}")
        check_fl = flr #or flourencense_identify(spectra,just_peaks=True) 
    else: check_fl=flourencense_identify(spectra,just_peaks=True)
    print(f"Flourense Tested on 2-> {check_fl}")         
    for drug in IDENTIFY_DRUG:
        # abort_if_requested(threadflag=True)
        # abort_if_requested()
        highest_peak= database_drugs[drug]['highest_drug_peak']
        out = get_pass_fail(spectra,drug,check_fl)
        convert ={}
        convert['drugname'] = drug
        convert['score'] = out['score']
        convert['y'] = list(spectra)
        convert['gs_peaks'] = out['golden_peaks']
        convert['range'] =  out['peaks_range']
        convert['peaks_found'] = out['peaks_found']
        # abort_if_requested(threadflag=True)
        # abort_if_requested()


        if len(highest_peak)==1:
            
            if np.argmax(out['y_processed'])+200 in range(highest_peak[0]-15,highest_peak[0]+15):
                convert['y_gs'] = list(normalize_to_peak(out['gold_sign'], highest_peak).values.ravel())
                convert['y_processed']= list(normalize_to_peak(out['y_processed'], highest_peak).values.ravel())
            else:
                convert['y_gs'] = list(normalize_to_peak(out['gold_sign'], highest_peak).values.ravel())
                convert['y_processed']= out['y_processed']
            
        elif len(highest_peak)==0:

            convert['y_gs']= out['gold_sign']
            convert['y_processed']= out['y_processed']

        frame.append(convert)

    df = pd.DataFrame(frame).sort_values(by='score',ascending=False)  
    top_two_drugs_past = df[df['score']>50].head(2).loc[:,'drugname'].values
    #print("Post old_result",top_two_drugs_past)
    past_df = df.copy()
    # abort_if_requested()
    # abort_if_requested(threadflag=True)
    def check_top_drug_result(df):
        in_top = False
        top_score = True if (df.head(2).loc[:,'score'].apply(lambda x: round(x,1))>=90).sum()==1 else  False
        for i in df[df['score']>=40].head(2).loc[:,'drugname'].values:
            print("C3 -> ",i)
            if i in NARCO_DRUG:
                in_top = True
        if in_top or top_score: return False
        return True
    
    df = df[df.apply(peaks_percentage,axis=1)>20] #20% of peaks_EXPERIMENT Values
    
    check_v3 = check_top_drug_result(df)

    if check_v3:
        #print("Check_V3")
        v3_data = peak_location_proba(spectra)
        top_two_drugs = [v3_data[0][0],v3_data[1][0]]
        #print("Current_result",top_two_drugs)
        if len(top_two_drugs_past)>0:
            if top_two_drugs[0]!=top_two_drugs_past[0]:
                for ind,narco_drug in enumerate(top_two_drugs):
                    if narco_drug in NARCO_DRUG: 
                        df = update_score(df,narco_drug,v3_data[ind][1])
                        #print("UPDATED",narco_drug)
        else:
            for ind,narco_drug in enumerate(top_two_drugs):
                    if narco_drug in NARCO_DRUG: 
                        df = update_score(df,narco_drug,v3_data[ind][1])
                        #print("UPDATED",narco_drug)
            
    df['score'] = df['score'].fillna(0)
    df['score'] = df['score'].apply(lambda x: round(x,3)) 
    # abort_if_requested()
    # abort_if_requested(threadflag=True)
    df=df[df['score']>=40]    
    possible_df= df[(df.score<50) & (df.score >=40)]
    df= df[df.score>=50]
    df = df.sort_values(by='score',ascending=False)
    endf = df.copy()

    if len(df)==0: 
        data = default_data
    else:
        d_name= select_drug_rdf(df)
        names = d_name
        checker_ex  = extreme_mapper(names)
        if checker_ex is not None: 
            fiver = peak_location_proba(spectra)
            fiver = list(map(lambda x: x[0],fiver[:5]))
            exp = expection_checker(spectra,names)
            if checker_ex in fiver or checker_ex in exp:
                d_name.append(checker_ex)
                names = d_name

        
        d_name = [i for i in names if not flourencense_identify(database_drugs[i]['xcal'],just_peaks=True)]
        if 'Sersheroin' in names: d_name.append('Sersheroin')
        if not check_fl: names = d_name
        print(f"Selected -> {d_name}")
        if len(d_name)>3:
            edf = ext_method(inp=spectra,d_name=d_name)
            names = filter_data_ext(edf)


        if len(names)!= len(df[df['drugname'].isin(names)]):df = endf[endf['drugname'].isin(names)].T.to_dict()
        else: df = df[df['drugname'].isin(names)].T.to_dict()
        possible_df_add= endf[~endf['drugname'].isin(names)]
        possible_df= pd.concat([possible_df,possible_df_add], axis=0)
        data = [df[key] for key in df.keys()]
        if check_fl1 and data[0]['score']<=55: 
            temp=[]
            temp.append(data[0])
            possible_df = pd.concat([possible_df,pd.DataFrame(temp)], axis=0)
            data = default_data   #threshold changed from 65 to 55
        if len(data)==0: data = default_data


        if len(data) > 1 and data[1]['drugname'] == 'Fentanyl':
            if data[1] and data[1]['score'] < 60:
                data = [data[0]]

    ##############################
    # abort_if_requested()
    # abort_if_requested(threadflag=True)
    if data and len(data)>1:
        print(
            f"Multiple drugs detected ({len(data)}).\n"
            f"Returning top 1 drug only  {data[0]['drugname']} (score={data[0]['score']})\n"
            f"Returning top 2 drug only  {data[1]['drugname']} (score={data[1]['score']})\n"
            f"top 1 drug only {data[0]['drugname']} (score={data[0]['score']}))"
        )
    data=[data[0]]
    if return_frame:
        return endf,edf
    return data
#######################################################################################


######################################### ID2 #####################################################
def id2get_ref_peaks(x,ref,flourense=False):
    prominence = 0.03
    pickup = 400 if flourense else 200
    if flourense:

        ref =  preprocess(ref,smoothing=True,idx=(400,1700))
    else :
        ref = preprocess(ref)
    g_peaks = find_peaks(ref,prominence=prominence)[0]+pickup
    g_peaks = g_peaks[(g_peaks >=400) & (g_peaks <= 1700)]
    
    # print("g_peaks --",g_peaks)
    pickups = 200 if len(x) else 350
    ref_peaks = find_all_near_peaks(find_peaks(x,prominence=prominence)[0]+pickups,g_peaks)
    # print("ref_peaks :- ",ref_peaks)
    return ref_peaks


def id2roi_processs(x,ref,size=100,flourense=True):
    pickup = 200
    xcopy = np.zeros_like(x)
    if flourense: 
        xt = preprocess(x,smoothing=True,idx=(400,1700),norm=True)
        xc = np.zeros_like(xcopy)
        xc[xt.index[0]-200:xt.index[-1]+1-200] = xt.values.copy()
        # print("-------")
        x = xc
    else: 
        x = preprocess(x)

    # print("x len",len(x))
    # print("xcopy len",len(xcopy))
    peak = id2get_ref_peaks(x,ref,flourense=flourense)
    # print("peak",peak)
    for p,ref in peak.items():
        if ref!=0:
            startref,endref = (ref-size-pickup,ref+(size+1)-pickup)
            # print("startref,endref",startref,endref)
            start,end = (p-size-pickup),p+(size+1)-pickup
            # print("start,end",start,end)
            # length = min(end - start, endref - startref)


            xcopy[start:end]=x[startref:endref].copy()
    return xcopy


def id2info_data_test(x,ref,flourense=False):
        xcopy = x.copy()
        pickup = 200 if len(xcopy)==2401 else 350
        if flourense: 
            pickup=400
            # x = pd.Series(x,index=np.arange(200,2601))
            # ref = pd.Series(ref,index=np.arange(200,2601))
            raw_norm = preprocess(x,smoothing=True,idx=(400,1700))
            ref_norm = preprocess(ref,smoothing=True,idx=(400,1700))

        else:
            raw_norm = preprocess(x)
            ref_norm = preprocess(ref)
        peaks =  find_peaks(raw_norm,prominence=prominence)[0] + pickup
        
        """dummy = pd.Series(xcopy,index=np.arange(200,2601))   
        peaks_values = dummy[peaks]   
        plt.scatter(peaks_values.index,peaks_values.values)
        plt.plot(dummy)"""
        tol_peaks =  find_peaks(ref_norm,prominence=prominence)[0] + pickup
        num_of_peaks,peaks_got,diff = number_of_peaks(peaks,tol_peaks)
        percentage = round((num_of_peaks/len(tol_peaks))*100,2)
        del x,raw_norm,xcopy
        return num_of_peaks,peaks_got,diff,percentage



def id2structure_score_check(x,ref,size=100,cal='xcal',flourense=False):
    #print(f"test on  = {drugname}")
    # global database_drugs
    # peaks = database_drugs[drugname]['peaks_list']   
    # ref = database_drugs[drugname]['raw']
    got_data = id2info_data_test(x,ref,flourense=flourense)

    if flourense:processed_scan = list(preprocess(x,smoothing=True).values)
    else: processed_scan = list(preprocess(x).values)

    if len(ref)>0:
        rop = id2roi_processs
        test_roi  = rop(x,ref,size=size,flourense=flourense)
        # print("-------ref drug trimimg ------------")
        ref_roi = rop(ref,ref,size=size,flourense=flourense)
        
        test = test_roi
        ref = ref_roi
        
        pvalue = Alert()
        ssim = round(pvalue.test(test,ref)[0]*100,3)
        corr = round(pearsonr(test, ref)[0] * 100, 3) if not math.isnan(pearsonr(test, ref)[0]) else 0
        peaks_found  = got_data[-1]
        percent = round(sum([ssim,corr,peaks_found])/3,3)
        return {"ssim":ssim,"corr":corr,"peaks":peaks_found,"percent":percent,"y_processed":processed_scan}
    else: 
        return 0
    

def id2compute_score_test(test,ref,check_intensity=True,flourense=False): 
    global database_drugs
    global prominence
    w_size=50  #experiment_1
    copydatat = pd.Series(np.zeros_like(test),index=np.arange(200,2601))
    copydatar = pd.Series(np.zeros_like(test),index=np.arange(200,2601))
    pickup = 200 if len(test)==2401 else 350
    if flourense: pickup = 200
    # test = pd.Series(test,index=np.arange(200,2601))  
    
    # g_peaks = database_drugs[drugname]['peaks_list']
    
    if flourense:
        # ref = pd.Series(database_drugs[drugname]['raw'],index=np.arange(200,2601))
        ref = preprocess(ref,smoothing=True,idx=(400,1700))
        copydatar.loc[ref.index[0]:ref.index[-1]] = ref.values.copy()
        ref = copydatar.copy()
        testp =  preprocess(test,smoothing=True,idx=(400,1700))
        copydatat.loc[testp.index[0]:testp.index[-1]]= testp.values.copy()
        testp = copydatat.copy()
        g_peaks = find_peaks(ref,prominence=prominence)[0]+pickup
        # print("g_peaks----:",g_peaks)
        ref = ref.values
        testp =  testp.values

    else:
        # ref = preprocess(database_drugs[drugname]['raw']).values
        g_peaks = find_peaks(preprocess(ref),prominence=prominence)[0]+pickup
        g_peaks = g_peaks[(g_peaks >=400) & (g_peaks <= 1700)]
        # print("g_peaks----:",g_peaks)
        ref = preprocess(ref).values
        testp =  preprocess(test).values
    
    processed_scan = list(preprocess(test).values)
   
    
    
    ref_peaks = find_all_near_peaks(find_peaks(ref,prominence=prominence)[0]+pickup,g_peaks)
    # print("ref_peaks -- :",ref_peaks)
    test_peaks = find_all_near_peaks(find_peaks(testp,prominence=prominence)[0]+pickup,g_peaks,multi=True,strict_mode=True)
    # print("test_peaks -- :",test_peaks)
    test_peaks_P = find_all_near_peaks(find_peaks(testp,prominence=prominence)[0]+pickup,g_peaks)
    
    # peaks_found = [i for i,j in test_peaks_P.items() if j!=0]
    ### ref peak shown if it is within 2 wavenumber else actual peak is displayed.
    shift_window=2
    peaks_found = [j if abs(i-j)> shift_window else i for i,j in test_peaks_P.items() if j!=0]
    
    peaks_found_score = []
    peak_con ={}  
    for i,j in ref_peaks.items():
        ref_part = ref[j-pickup-w_size:j-pickup+w_size]
        maxd=0
        maxp = 0
        if len(test_peaks[i])>0:
            for k in test_peaks[i]:
                test_part = testp[k-pickup-w_size:k-pickup+w_size]
                ins = abs(max(test_part)-max(ref_part))
                pvalue = Alert()
                score = round(pvalue.test(test_part,ref_part)[0]*100,3)
                score = score * dynamic_score(int(abs(k-i)))
                
                #experiment_3 to check on corr in place of ssim
                #score = round(pearsonr(test_part,ref_part)[0]*100,3)
                
                if maxd<score: 
                    maxd = score
                    maxp = k
            if check_intensity:
                if maxp!=0:
                    test_part_peak_int = testp[maxp-pickup]
                    ref_part_peaks_int = ref[j-pickup]
                    diff_intensity = round(abs(test_part_peak_int - ref_part_peaks_int),2)
                    score_intensity = dynamic_score(diff_intensity,mode='intensity')*100
                    peak_con[i]=np.mean((maxd,score_intensity))
                else: peak_con[i]=0
            else:
                peak_con[i]=maxd
        if maxp!=0: 
            peaks_found_score.append(maxp)
    
    sum_of_con = round(sum([val for key,val in peak_con.items()])/len(g_peaks),3)
    return {"score":sum_of_con,"peaks_found":peaks_found,"y_processed":processed_scan,"ref_peaks":ref_peaks,"test_peaks":test_peaks,"frame":peak_con}

def range_info(lst):
    return (round(min(lst) - 100, 1), round(max(lst) + 100, 1))


def id2peak_score_check(sample,ref,flourense=False):
    pickup = 400 if flourense else 200
    if flourense:
        ref_process = preprocess(ref, smoothing=True, idx=(400,1700))
        sample_process = preprocess(sample, smoothing=True, idx=(400,1700))
    else:
        sample_process = preprocess(sample)
        ref_process = preprocess(ref)

    sample_peaks= find_peaks(sample_process, prominence=0.03)[0]+pickup
    sample_peaks = sample_peaks[(sample_peaks >=400) & (sample_peaks <= 1700)]

    ref_peaks= find_peaks(ref_process, prominence=0.03)[0]+pickup
    ref_peaks = ref_peaks[(ref_peaks >=400) & (ref_peaks <= 1700)]
    c = 0
    matching_value = []
    for i,j in find_all_near_peaks(sample_peaks,ref_peaks,multi=True,strict_mode=True).items():
        if len(j)>0 and abs(i-j[0])<=4: 
            matching_value.append(j[0])
            c+=1
    score =  round(c/max(len(sample_peaks),len(ref_peaks)) * 100, 2)
    return {"score":score, "matching_peaks":matching_value,"ref_peaks":ref_peaks}

def score_weightage(peak_score, structure_score, compute_score):
    w1 = 2  # Weight for peak_score
    w2 = 1  # Weight for structure_score
    w3 = 1  # Weight for compute_score

    weighted_avg_score = (
        (w1 * peak_score) + (w2 * structure_score) + (w3 * compute_score)
    ) / (w1 + w2 + w3)
    # print("weighted_avg_score",weighted_avg_score)
    return weighted_avg_score

def id2peak_top_peaks(sample,ref,flourense=False):
    pickup = 400 if flourense else 200
    if flourense:
        ref_process = preprocess(ref, smoothing=True, idx=(400,1700))
        sample_process = preprocess(sample, smoothing=True, idx=(400,1700))
    else:
        sample_process = preprocess(sample)
        ref_process = preprocess(ref)

    sample_peaks= find_peaks(sample_process, prominence=0.03)[0]+pickup
    ref_peaks= find_peaks(ref_process, prominence=0.03)[0]+pickup    
    if len(ref_peaks)>10:
        x = ref_process.index.to_numpy(dtype=float)   
        y = ref_process.values                        
        peaks, _ = find_peaks(y,prominence=0.03)
        prominences = peak_prominences(y, peaks)[0]
        # Get top 10 by prominence
        top_indices = np.argsort(prominences)[-10:][::-1]
        top_peaks = peaks[top_indices]
        ref_peaks = top_peaks+pickup
    c = 0
    matching_value = []
    for i,j in find_all_near_peaks(sample_peaks,ref_peaks,multi=True,strict_mode=True).items():
        if len(j)>0 and abs(i-j[0])<=4: 
            matching_value.append(int(j[0]))
            c+=1
    return {"matching_peaks":matching_value,"ref_peaks":ref_peaks.tolist()}

def ID2_Module(sample_spectra):
    '''This module iterate over all spectra scans stored in the ID2 database and compare each scan with a given sample
      spectrum. The goal is to identify the best match for the sample spectrum based on scoring criteria such 
      as peak alignment, structure similarity, and computed matching scores.'''
    frame = []
    # print('checking type',type(sample_spectra))
    if type(sample_spectra) == list:
            sample_spectra = pd.Series(sample_spectra,index=np.arange(200,2601)) 
    row_count = ID2_DB_DATA.get_row_count()
    for row in range(1,row_count+1):
        id2_data = ID2_DB_DATA.fetch_data_by_id(record_id=row)
        convert = {}
        convert["result"] =  id2_data[0]['feedback']
        convert["id"] = id2_data[0]['Scan_id']   
        ref_spectra = pd.Series(eval(id2_data[0]['xcal']),index=np.arange(200,2601))           
        # Identify fluorescence
        samplefl = flourencense_identify(sample_spectra)
        ref_fl = flourencense_identify(ref_spectra)
        flourense = samplefl or ref_fl
        
        # Check peak score
        peak_score = id2peak_score_check(sample_spectra, ref_spectra, flourense=flourense)
        peaks = id2peak_top_peaks(sample_spectra, ref_spectra, flourense=flourense)
        convert["gs_peaks"] = peaks['ref_peaks']
        convert["peaks_found"] =peaks['matching_peaks']
        convert['range'] = range_info(np.array(peak_score['ref_peaks']).tolist())
        convert["peaks_score"] = peak_score['score']
        
        structure = id2structure_score_check(sample_spectra, ref_spectra, flourense=flourense)
        convert["structure_score"] = structure['percent']
        
        compute = id2compute_score_test(sample_spectra, ref_spectra, flourense=flourense)
        convert["compute_score"] = compute['score']
        
        # ID_SCORE2 = round((peak_score['score']+ structure['percent'] + compute['score']) / 3, 3)
        ID_SCORE = score_weightage(peak_score=peak_score['score'],structure_score=structure['percent'],compute_score=compute['score'])
        
        convert["ID_SCORE"] = ID_SCORE
        # Append the record to the frame only if both checks passed
        if ID_SCORE <= 75 :
             continue   
        convert['y'] = list(sample_spectra.values)
        convert["y_gs"] =  list(preprocess(ref_spectra).values)
        convert["y_processed"] = list(preprocess(sample_spectra).values)
        # Append the record to the frame only if both checks passed
        frame.append(convert)
    
    # Create DataFrame from the frame
    if not frame:
        result =[{
            "drugname" : "Unknown",
            "score" : 0 ,
            "sample" : sample_spectra
        }]
        return result
    sorted_frame = sorted(frame, key=lambda x: x["ID_SCORE"], reverse=True)
    # Get the top record
    top_result = sorted_frame[0]

    return [{
        "drugname": top_result["result"],
        "score" : top_result['ID_SCORE'],
        "Scanid_to_matched" : top_result["id"],
        "gs_peaks": top_result["gs_peaks"],
        "range" : top_result["range"],
        "peaks_found": top_result["peaks_found"],
        "y" : top_result["y"],
        "y_gs": top_result["y_gs"],
        "y_processed": top_result["y_processed"]
    }]



########################################### ID3 DRUG######################
################################# Functions for ID3(Easy ER Logic) ####################################

'''Return dataframe where score > 30 to select top two drugs for identification'''

def identify_proba_ID3(spectra,flr=False,return_frame=False):
    check_v3 = False
    names,edf=0,0
    default_data = [{
                    'drugname':"Unknown",
                    "score":0,
                    "y":list(spectra),
                    'gs_peaks':[],
                    "range":(200,2601),
                    "peaks_found": top_intensity_peaks(spectra),#[int(i)+200 for i in list(find_peaks(list(preprocess(spectra).values),prominence=0.6)[0])[:7]],
                    "y_gs":[],
                    "y_processed":list(preprocess(spectra).values)      
                }]

    frame  = []
    def update_score(df,drug,score):
        if isinstance(df,pd.DataFrame):
            df.loc[df['drugname']==drug,'score']=score
        return df 
    def peaks_percentage(df):
        pf = len(df['peaks_found'])
        dpf = len(database_drugs[df['drugname']]['peaks_list'])
        if dpf>2:dk = round((pf/dpf) * 100,3)
        else: dk = 40
        return dk
    check_fl1 = flourencense_identify(spectra)
    #print(f"Flourense Tested on 1-> {check_fl1}") 
    
    if flr==True: 
        #print(f"From Camera Shutter -> {flr}")
        check_fl = flr #or flourencense_identify(spectra,just_peaks=True) 
    else: check_fl=flourencense_identify(spectra,just_peaks=True)
    #print(f"Flourense Tested on 2-> {check_fl}")         
    for drug in IDENTIFY_DRUG:
        highest_peak= database_drugs[drug]['highest_drug_peak']
        out = get_pass_fail(spectra,drug,check_fl)
        convert ={}
        convert['drugname'] = drug
        convert['score'] = out['score']
        convert['y'] = list(spectra)
        convert['gs_peaks'] = out['golden_peaks']
        convert['range'] =  out['peaks_range']
        convert['peaks_found'] = out['peaks_found']

        if len(highest_peak)==1:
            
            if np.argmax(out['y_processed'])+200 in range(highest_peak[0]-15,highest_peak[0]+15):
                convert['y_gs'] = list(normalize_to_peak(out['gold_sign'], highest_peak).values.ravel())
                convert['y_processed']= list(normalize_to_peak(out['y_processed'], highest_peak).values.ravel())
            else:
                convert['y_gs'] = list(normalize_to_peak(out['gold_sign'], highest_peak).values.ravel())
                convert['y_processed']= out['y_processed']
            
        elif len(highest_peak)==0:

            convert['y_gs']= out['gold_sign']
            convert['y_processed']= out['y_processed']

        frame.append(convert)

    df = pd.DataFrame(frame).sort_values(by='score',ascending=False)  
    top_two_drugs_past = df[df['score']>50].head(2).loc[:,'drugname'].values
    #print("Post old_result",top_two_drugs_past)
    past_df = df.copy()

    def check_top_drug_result(df):
        in_top = False
        top_score = True if (df.head(2).loc[:,'score'].apply(lambda x: round(x,1))>=90).sum()==1 else  False
        for i in df[df['score']>=40].head(2).loc[:,'drugname'].values:
            #print("C3 -> ",i)
            if i in NARCO_DRUG:
                in_top = True
        if in_top or top_score: return False
        return True
    
    df = df[df.apply(peaks_percentage,axis=1)>20] #20% of peaks_EXPERIMENT Values
    
    check_v3 = check_top_drug_result(df)

    if check_v3:
        #print("Check_V3")
        v3_data = peak_location_proba(spectra)
        top_two_drugs = [v3_data[0][0],v3_data[1][0]]
        #print("Current_result",top_two_drugs)
        if len(top_two_drugs_past)>0:
            if top_two_drugs[0]!=top_two_drugs_past[0]:
                for ind,narco_drug in enumerate(top_two_drugs):
                    if narco_drug in NARCO_DRUG: 
                        df = update_score(df,narco_drug,v3_data[ind][1])
                        #print("UPDATED",narco_drug)
        else:
            for ind,narco_drug in enumerate(top_two_drugs):
                    if narco_drug in NARCO_DRUG: 
                        df = update_score(df,narco_drug,v3_data[ind][1])
                        #print("UPDATED",narco_drug)
            
    df['score'] = df['score'].fillna(0)
    df['score'] = df['score'].apply(lambda x: round(x,3)) 

    df = df[df['score'] >= 30]    
    df = df.sort_values(by='score', ascending=False)

    easyer_df = df.copy()
    return easyer_df


class ScanSegreggationByType(object):

    class Smoother(object):
        
        def __init__(self, Y, smoothness_param, deriv_order=1):
            self.y = Y
            assert deriv_order > 0, 'deriv_order must be an int > 0'
            d = np.zeros(deriv_order * 2 + 1, dtype=int)
            d[deriv_order] = 1
            d = np.diff(d, deriv_order)
            n = self.y.shape[0]
            k = len(d)
            s = float(smoothness_param)
            diag_sums = np.vstack([
                np.pad(s * np.cumsum(d[-i:] * d[:i]), ((k - i, 0),), 'constant')
                for i in range(1, k + 1)
            ])
            upper_bands = np.tile(diag_sums[:, -1:], n)
            upper_bands[:, :k] = diag_sums
            for i, ds in enumerate(diag_sums):
                upper_bands[i, -i - 1:] = ds[::-1][:i + 1]
            self.upper_bands = upper_bands

        def smooth(self, w):
            foo = self.upper_bands.copy()
            foo[-1] += w  
            return solveh_banded(foo, w * self.y, overwrite_ab=True, overwrite_b=True)

    def als_baseline(self, intensities, asymmetry_param=0.05, smoothness_param=1e6,
                     max_iters=10, conv_thresh=1e-5, verbose=False):
        '''Perform asymmetric least squares baseline removal.'''
        smoother = self.Smoother(intensities, smoothness_param, deriv_order=2)
        p = asymmetry_param
        w = np.ones(intensities.shape[0])
        for i in range(max_iters):
            z = smoother.smooth(w)
            mask = intensities > z
            new_w = p * mask + (1 - p) * (~mask)
            conv = np.linalg.norm(new_w - w)
            if verbose:
                print(f"Iteration {i + 1}: Convergence {conv}")
            if conv < conv_thresh:
                break
            w = new_w
        else:
            print(f'ALS did not converge in {max_iters} iterations')
        return z

    def Baseline_Reduction(self, df):
        baseline = ScanSegreggationByType()
        for i in range(len(df)):
            baseline_temp = baseline.als_baseline(df.iloc[i].values.ravel())
            df.iloc[i] = df.iloc[i] - baseline_temp
        return df

    def Smoothing(self, df):
        for i in range(len(df)):
            df.iloc[i] = butterworth_filter(np.array(df.iloc[i]))
        return df

    def peak_normalization(self, df):
        if len(df) > 0:
            df = np.array(df)
            if len(df) == 2401:
                x = df / (df.max() - df.min())
                return x
            if len(df) == 1802:
                x = df / (df[0:1500].max() - df.min())
                return x

    def flat_scan_check(self, filepath):
        df = pd.DataFrame(filepath).T
        df.columns = range(200, 200 + df.shape[1])
        bs = self.Baseline_Reduction(df)
        smoothed_data = self.Smoothing(bs)
        normalized_data = self.peak_normalization(smoothed_data.T.values.ravel())
        peaks, _ = find_peaks(normalized_data, prominence=0.03)
        list_peaks = [i for i in peaks if 600 <= i + 200 <= 1400]
        return not list_peaks


def check_good_spectra_condition(spectra):
    try:
        if ScanSegreggationByType().flat_scan_check(spectra):
            return False
        elif flourencense_identify(spectra):
            return False
        else:
            # Spectra is a good spectrum
            return True
    except (ValueError, SyntaxError) as e:
        print(f"Error processing spectra: {e}")
        return False
    
    
def calculate_peak_differences(ref, drugname):
    try:
        most_repeated_difference = None
        ref_peaks_dict = get_ref_peaks(ref, drugname)
        valid_pairs = [(key, value) for key, value in ref_peaks_dict.items()]
        peak_differences = [abs(key - value) for key, value in valid_pairs]

        if peak_differences:
            frequency_counts = {num: peak_differences.count(num) for num in set(peak_differences)}
            max_frequency = max(frequency_counts.values())
            most_frequent_numbers = [num for num, freq in frequency_counts.items() if freq == max_frequency]
            most_repeated_difference = max(most_frequent_numbers)

            occurrence = peak_differences.count(most_repeated_difference)
            total_items = len(peak_differences)
            percentage = round((occurrence / total_items) * 100,3)
            print('peak percentage',percentage)

            # Determine consistency based on percentage
            is_consistent = percentage >= 60
        else:
            most_repeated_difference = None
            percentage = 0
            is_consistent = False

        return {
            "peak_differences": peak_differences,
            "is_consistent": is_consistent,
            "most_repeated_difference": most_repeated_difference,
            "percentage_occurrence": percentage
        }
    except Exception as e:
        return {"error": str(e)}
    
def code1(spectra, easyer_df):
    try:
        if easyer_df.empty:
            return {'No records in df present'}
        
        top_drugs = easyer_df.head(2)['drugname'].values
        print('top two drugs', top_drugs)

        peak_diff_1 = calculate_peak_differences(spectra, top_drugs[0])
        peak_diff_2 = None
        if len(top_drugs) > 1:
            peak_diff_2 = calculate_peak_differences(spectra, top_drugs[1])

        structure_result_1 = structure_score(spectra, top_drugs[0])
        structure_result_2 = structure_score(spectra, top_drugs[1]) if len(top_drugs) > 1 else {'percent': 0}
        
        structure_score_1 = structure_result_1.get('ssim', 0)
        structure_score_2 = structure_result_2.get('ssim', 0)
        
        selected_drug = None
        if peak_diff_1.get('is_consistent', False):
            selected_drug = top_drugs[0]             
        elif peak_diff_2 and peak_diff_2.get('is_consistent', False):
            selected_drug = top_drugs[1]            
        elif max(structure_score_1, structure_score_2) >= 90:
            selected_drug = top_drugs[0] if structure_score_1 > structure_score_2 else top_drugs[1]
        else:
            return {'found': 'Unknown'}  # No peak consistency and no high structure score

        print('structure score c1', structure_result_1['ssim'], structure_result_2['ssim'])

        if selected_drug == top_drugs[0]:
            percentage_occurrence = peak_diff_1.get('percentage_occurrence', 0)
            final_structure_score = structure_score_1
        else:
            percentage_occurrence = peak_diff_2.get('percentage_occurrence', 0)
            final_structure_score = structure_score_2
        
        final_score = (percentage_occurrence + final_structure_score) / 2

        # Check threshold conditions for a valid match
        if final_structure_score >= 90:
            found_name = selected_drug
            selected_row = easyer_df[easyer_df['drugname'] == found_name]
            
            peaks_found = selected_row['peaks_found'].values[0]
            peaks_range = selected_row['range'].values[0]
            gold_sign = selected_row['y_gs'].values[0]
            golden_peaks = selected_row['gs_peaks'].values[0]
            y_processed = selected_row['y_processed'].values[0]
            
            # Convert the entire row to a list of dictionaries
            data_dict = selected_row.to_dict(orient='records')

            return {
                "found": found_name,
                "score": final_score,
                "gold_sign": gold_sign,
                "peaks_found": peaks_found,
                "peaks_range": peaks_range,
                "golden_peaks": golden_peaks,
                "y_processed": y_processed,
                "identify_result": data_dict
            }

    except KeyError as e:
        return {"error": f"KeyError: {str(e)}"}
    except ValueError as e:
        return {"error": f"ValueError: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

    return {'found': 'Unknown'}



def info_data_C2_ID3(x,drug,flourense=False):
    prominence = 0.02
    global database_drugs
    if isinstance(x,pd.DataFrame):
        l=[]
        for i in range(x.shape[0]):
            d={}
            y =  x.iloc[i]
            raw_norm = preprocess(y)
            pickup = 200 if len(raw_norm)==2401 else 350
            peaks =  find_peaks(raw_norm,prominence=prominence)[0] + pickup
            tol_peaks=database_drugs[drug]['peaks_list']
            num_of_peaks,peaks_got,diff = number_of_peaks(peaks,tol_peaks)
            d['id']=i
            d['Num']=num_of_peaks
            d['peaks']=peaks_got
            d['diff']=diff
            d['percentage'] = round((num_of_peaks/len(tol_peaks))*100,2)
            del x,raw_norm,xcopy
            l.append(d) 
        return pd.DataFrame(l)
    if isinstance(x,pd.Series) or isinstance(x,list) or isinstance(x,np.ndarray):
            xcopy = x.copy()
            pickup = 200 if len(xcopy)==2401 else 350
            if flourense: 
                pickup=400
                x = pd.Series(x,index=np.arange(200,2601))
                raw_norm = preprocess(x,smoothing=True,idx=(400,1700))
            else:
                raw_norm = preprocess(x)
            peaks =  find_peaks(raw_norm,prominence=prominence)[0] + pickup
            """dummy = pd.Series(xcopy,index=np.arange(200,2601))   
            peaks_values = dummy[peaks]   
            plt.scatter(peaks_values.index,peaks_values.values)
            plt.plot(dummy)"""
            tol_peaks=database_drugs[drug]['peaks_list']
            num_of_peaks,peaks_got,diff = number_of_peaks(peaks,tol_peaks)
            percentage = round((num_of_peaks/len(tol_peaks))*100,2)
            del x,raw_norm,xcopy
            return num_of_peaks,peaks_got,diff,percentage

def structure_score_C2_ID3(x,drugname,size=100,cal='xcal',flourense=False):
    #print(f"test on  = {drugname}")
    global database_drugs
    peaks = database_drugs[drugname]['peaks_list']   
    ref = database_drugs[drugname]['raw']
    got_data = info_data_C2_ID3(x,drugname,flourense=flourense)

    if flourense:processed_scan = list(preprocess(x,smoothing=True).values)
    else: processed_scan = list(preprocess(x).values)

    if len(ref)>0:
        if drugname == "Lidocaine": rop =roi_process_2
        else: rop = roi_process
        test_roi  = rop(x,drugname,cal=cal,size=size,flourense=flourense)
        ref_roi = rop(ref,drugname,cal=cal,size=size,flourense=flourense)
        
        test = test_roi
        ref = ref_roi
        
        pvalue = Alert()
        ssim = round(pvalue.test(test,ref)[0]*100,3)
        corr = round(pearsonr(test, ref)[0] * 100, 3) if not math.isnan(pearsonr(test, ref)[0]) else 0
        peaks_found  = got_data[-1]
        percent = round(sum([ssim,corr,peaks_found])/3,3)
        return {"ssim":ssim,"corr":corr,"peaks":peaks_found,"percent":percent,"y_processed":processed_scan}
    else: 
        return 0
    
def compute_score_C2_ID3(test,drugname=None,check_intensity=True,flourense=False): 
    global database_drugs
    prominence = 0.02
    w_size=50  #experiment_1
    copydatat = pd.Series(np.zeros_like(test),index=np.arange(200,2601))
    copydatar = pd.Series(np.zeros_like(test),index=np.arange(200,2601))
    pickup = 200 if len(test)==2401 else 350
    if flourense: pickup = 200
    test = pd.Series(test,index=np.arange(200,2601))  
    
    g_peaks = database_drugs[drugname]['peaks_list']
    
    if flourense:
        ref = pd.Series(database_drugs[drugname]['raw'],index=np.arange(200,2601))
        ref = preprocess(ref,smoothing=True,idx=(400,1700))
        copydatar.loc[ref.index[0]:ref.index[-1]] = ref.values.copy()
        ref = copydatar.copy()
        testp =  preprocess(test,smoothing=True,idx=(400,1700))
        copydatat.loc[testp.index[0]:testp.index[-1]]= testp.values.copy()
        testp = copydatat.copy()

        ref = ref.values
        testp =  testp.values

    else:
        ref = preprocess(database_drugs[drugname]['raw']).values
        testp =  preprocess(test).values
    
    processed_scan = list(preprocess(test).values)
   
    ref_peaks = find_all_near_peaks(find_peaks(ref,prominence=prominence)[0]+pickup,g_peaks)
    test_peaks = find_all_near_peaks(find_peaks(testp,prominence=prominence)[0]+pickup,g_peaks,multi=True,strict_mode=True)
    test_peaks_P = find_all_near_peaks(find_peaks(testp,prominence=prominence)[0]+pickup,g_peaks)
    
    # peaks_found = [i for i,j in test_peaks_P.items() if j!=0]
    ### ref peak shown if it is within 2 wavenumber else actual peak is displayed.
    shift_window=2
    peaks_found = [j if abs(i-j)> shift_window else i for i,j in test_peaks_P.items() if j!=0]
    
    peaks_found_score = []
    peak_con ={}  
    for i,j in ref_peaks.items():
        ref_part = ref[j-pickup-w_size:j-pickup+w_size]
        maxd=0
        maxp = 0
        if len(test_peaks[i])>0:
            for k in test_peaks[i]:
                test_part = testp[k-pickup-w_size:k-pickup+w_size]
                ins = abs(max(test_part)-max(ref_part))
                pvalue = Alert()
                score = round(pvalue.test(test_part,ref_part)[0]*100,3)
                score = score * dynamic_score(int(abs(k-i)))
                
                #experiment_3 to check on corr in place of ssim
                #score = round(pearsonr(test_part,ref_part)[0]*100,3)
                
                if maxd<score: 
                    maxd = score
                    maxp = k
            if check_intensity:
                if maxp!=0:
                    test_part_peak_int = testp[maxp-pickup]
                    ref_part_peaks_int = ref[j-pickup]
                    diff_intensity = round(abs(test_part_peak_int - ref_part_peaks_int),2)
                    score_intensity = dynamic_score(diff_intensity,mode='intensity')*100
                    peak_con[i]=np.mean((maxd,score_intensity))
                else: peak_con[i]=0
            else:
                peak_con[i]=maxd
        if maxp!=0: 
            peaks_found_score.append(maxp)
    
    sum_of_con = round(sum([val for key,val in peak_con.items()])/len(g_peaks),3)
    return {"score":sum_of_con,"peaks_found":peaks_found,"y_processed":processed_scan,"ref_peaks":ref_peaks,"test_peaks":test_peaks,"frame":peak_con}


def code2(spectra, easyer_df):
    try:
        if easyer_df.empty:
            return {'No records in df present'}
        
        top_drugs = easyer_df.head(2)

        best_drug = None
        max_structure_score = 0
        max_compute_score = 0

        # Step 3: Evaluate the selected drugs
        for _, row in top_drugs.iterrows():
            drug_name = row['drugname']

            # Evaluate structure score
            structure_result = structure_score_C2_ID3(spectra, drug_name)
            structure_percent = structure_result.get('percent', 0)

            # Evaluate compute score
            compute_result = compute_score_C2_ID3(spectra, drug_name)
            compute_percent = compute_result.get('score', 0)

            # Compare and select the best drug based on the conditions
            if structure_percent > max_structure_score and compute_percent > max_compute_score:
                best_drug = drug_name
                max_structure_score = structure_percent
                max_compute_score = compute_percent

        # Step 4: Check if the best drug meets the threshold criteria
        print('c2 structure score', max_structure_score)
        print('c2 compute score', max_compute_score)
        if max_structure_score >= 60 and max_compute_score >= 60:
            selected_drug = best_drug

            final_score = (max_structure_score + max_compute_score) / 2
           
            # Extract details for the selected drug
            selected_row = easyer_df[easyer_df['drugname'] == selected_drug]
            
            peaks_found = selected_row['peaks_found'].values[0]
            peaks_range = selected_row['range'].values[0]
            gold_sign = selected_row['y_gs'].values[0]
            golden_peaks = selected_row['gs_peaks'].values[0]
            y_processed = selected_row['y_processed'].values[0]
            
            # Convert the entire row to a list of dictionaries
            data_dict = selected_row.to_dict(orient='records')

            return {
                "found": selected_drug,
                "score": final_score,
                "gold_sign": gold_sign,
                "peaks_found": peaks_found,
                "peaks_range": peaks_range,
                "golden_peaks": golden_peaks,
                "y_processed": y_processed,
                "identify_result": data_dict
            }

    except KeyError as e:
        return {"error": f"KeyError: {str(e)}"}
    except ValueError as e:
        return {"error": f"ValueError: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

    return {'found': 'Unknown'}


################################## ID3 integrated with ID1 ############################################
def identify_drug(spectra, flr=False):
    try:
        # from oak_utilities import abort_if_requested
        # abort_if_requested()
        # abort_if_requested(threadflag=True)
        print("Running Normal identification with out ID2 and ID3 for advance algorithm")
        data = identify_proba(spectra, flr=flr)
        if data[0]['drugname'] == 'Unknown':
            if check_good_spectra_condition(spectra):
                easyer_df = identify_proba_ID3(spectra, flr=flr, return_frame=True)
                max_score = easyer_df['score'].max() if not easyer_df.empty else 0
                if max_score > 30:
                    print('consistency & score cond satisfied')
                    code1_result = code1(spectra,easyer_df) #code1 description (change)
                    #print('in code1')
                    if code1_result['found'] != 'Unknown' and code1_result['found'] in NARCO_DRUGS_LIST:
                        print('code1 cond satisfied')
                        return code1_result
                    
                    # If Code1 result is also unknown, proceed to Code2
                    print('code2 start')
                    code2_result = code2(spectra,easyer_df)
                    #print('in code2')   #code2 description (change)
                    if code2_result['found'] != 'Unknown' and code2_result['found'] in NARCO_DRUGS_LIST:
                        print('code2 cond satisfied')
                        return code2_result

                    # If both Code1 and Code2 fail
                    print('code1 & code2 cond fail')
                    pass
                else:
                    print('not satisfied good & score cond')
                    pass
            
           
        # abort_if_requested()
        # abort_if_requested(threadflag=True)   
        found_name= data[0]['drugname']
        score = data[0]['score']
        peaks_found = data[0]['peaks_found']
        peaks_range = data[0]["range"]
        gold_sign = data[0]['y_gs']
        golden_peaks= data[0]['gs_peaks']
        y_processed = data[0]['y_processed']
        if found_name == 'None' or score == 0: found_name = 'Unknown'
        
    
    except Exception as e:
        print(f"Failed Identification test: {e}")
        found_name = 'Unknown'
        score = 0
        peaks_found = [int(i) + 200 for i in list(find_peaks(list(preprocess(spectra).values), prominence=0.3)[0])][:5]
        peaks_range = (200, 2601) if len(spectra) == 2401 else (350, 2152)
        gold_sign = []
        golden_peaks = []
        data = []
        y_processed = list(preprocess(spectra).values)
        # possible_match = []

    # Prepare and return final data
    data_to_return = {
        "found": found_name,
        "score": score,
        "gold_sign": gold_sign,
        "peaks_found": peaks_found,
        "peaks_range": peaks_range,
        "golden_peaks": golden_peaks,
        'y_processed': y_processed,
        "identify_result": data
        # "possible_match": possible_match
    }
    return data_to_return

###########################Q Identification##############################
   
def q_identify_pipeline(x: list,drugname: str,solvents: list=['Water']):
    """
    quatification_identification perform quantification 
    """
    
    if str(drugname).lower() == "lidocaine": solvents=["Dextrose"]
    def q_scoring(x,y):

        def bin_check(x,y,size=20):
            import math
            l = []
            for i in range(0,len(x),size):
                if (i+size)<len(x):
                    part_x = x[i:i+size]
                    part_y =  y[i:i+size]
                    kl = pearsonr(part_x,part_y)[0]
                    if math.isnan(float(kl)):
                        kl=0
                    l.append(kl)
            return np.mean(l)
        def q_scoring_preprocess(x):
            x=x[200:1400]
            #x = butterworth_filter(x)
            da = np.gradient(x)
            for _ in range(1):x = np.gradient(da)
            x = butterworth_filter(x)
            x = x-als_baseline(x)
            #x =  np.where(x>0,x,0)
            #x = peak_normalization(x)
            x = (x-min(x))/(max(x)-min(x))
            return x
    
        a = Alert()
        
        data_x = np.array(q_scoring_preprocess(x))
        data_y = np.array(q_scoring_preprocess(y))

        d={}
        d['Alert'] = a.test(data_x,data_y)[0]
        corr = pearsonr(data_x,data_y)[0]
        d['Corr'] = corr if corr>0 else 0
        bin_score = bin_check(data_x,data_y,size=50)
        d['bin_score'] = bin_score if bin_score>0 else 0
        knn = KNeighborsRegressor(n_neighbors=3)
        knn.fit((data_x).reshape(-1, 1),(data_y).reshape(-1, 1))
        knn_score =knn.score((data_x).reshape(-1, 1),(data_y).reshape(-1, 1))
        d['knn'] = knn_score if knn_score>0 else 0
        d['score'] = sum(d.values())/len(d)
        return d
    
    drug_report = Drug_Detect_Report(database=database_drugs,drug=drugname,solvents=solvents)
    data_of_report = drug_report.drug_confidence(x)
    
    local_db = drug_report.db
    ref = local_db['raw'] #xcal
    xnorm = peak_normalization(x)
     
    score_report = q_scoring(xnorm,ref)
    
    data_of_report = {**data_of_report,**score_report}
    
    if data_of_report['score']>local_db['pass_threshold']: data_of_report['alert'] = 'low'
    elif data_of_report['score']>0.45 and data_of_report['score']<local_db['pass_threshold']: data_of_report['alert'] = "medium"
    else: data_of_report['alert'] = 'high'
        
    if data_of_report['alert']=='high': 
        data_of_report['found_drug'] = list(map(lambda x: x.lower(), data_of_report['found_drug']))
        try:
            data_of_report['found_drug'].pop(data_of_report['found_drug'].index(str(drugname).lower()))
            data_of_report['found_drug'] = list(map(lambda x: x.capitalize(), data_of_report['found_drug']))
        except Exception as e: print(e)
        data_of_report['not_found_drug'].append(drugname)
    
    data_of_report['not_found_drug'] = list(set(data_of_report['not_found_drug']))
    data_of_report['found_drug'] = list(set(data_of_report['found_drug']))
    return data_of_report 

###########################Q Identification_Syringe ##############################
def q_identify_pipeline_syringe(x: list,drugname: str,solvents: list=['Water']):
    """
    quatification_identification perform quantification 
    """
    
    if str(drugname).lower() == "lidocaine": solvents=["Dextrose"]
    def q_scoring(x,y):

        def bin_check(x,y,size=20):
            import math
            l = []
            for i in range(0,len(x),size):
                if (i+size)<len(x):
                    part_x = x[i:i+size]
                    part_y =  y[i:i+size]
                    kl = pearsonr(part_x,part_y)[0]
                    if math.isnan(float(kl)):
                        kl=0
                    l.append(kl)
            return np.mean(l)
        def q_scoring_preprocess(x):
            x=x[200:1400]
            #x = butterworth_filter(x)
            da = np.gradient(x)
            for _ in range(1):x = np.gradient(da)
            x = butterworth_filter(x)
            x = x-als_baseline(x)
            #x =  np.where(x>0,x,0)
            #x = peak_normalization(x)
            x = (x-min(x))/(max(x)-min(x))
            return x
    
        a = Alert()
        
        data_x = np.array(q_scoring_preprocess(x))
        data_y = np.array(q_scoring_preprocess(y))

        d={}
        d['Alert'] = a.test(data_x,data_y)[0]
#         corr = pearsonr(data_x,data_y)[0]
#         d['Corr'] = corr if corr>0 else 0
#         bin_score = bin_check(data_x,data_y,size=50)
#         d['bin_score'] = bin_score if bin_score>0 else 0
#         knn = KNeighborsRegressor(n_neighbors=3)
#         knn.fit((data_x).reshape(-1, 1),(data_y).reshape(-1, 1))
#         knn_score =knn.score((data_x).reshape(-1, 1),(data_y).reshape(-1, 1))
#         d['knn'] = knn_score if knn_score>0 else 0
        d['score'] = sum(d.values())/len(d)
        return d

    def syr_peaks_check(peaks,already_known_peaks,window_size): 
        
        pc_and_pp_peaks=[353,  366,  371,  396,  401,  455,  487,  491,  528,  547,  578,  583,  637,  706,
                        735,  809,  831,  842,  891,  903,  923,  974, 1000, 1007, 1040, 1115, 1155, 1183,
                        1224, 1241, 1261, 1314, 1332, 1365, 1438, 1462, 1468, 1605, 1632, 1774, 1820, 1920,
                        1979, 2057, 2078, 2114]
        
        known_peaks=[]
        unknown_peaks = []

        for peak in peaks:
            found = False

            for pc_value in pc_and_pp_peaks:
                if abs(peak - pc_value) <= window_size:
                    known_peaks.append(peak)
                    found = True
                    break 

            if not found:
                unknown_peaks.append(peak)
        s={}
        s['known_peaks'] = known_peaks + already_known_peaks
        s['unknown']=unknown_peaks
        
        return s
    
    drug_report = Drug_Detect_Report(database=database_drugs,drug=drugname,solvents=solvents)
    data_of_report = drug_report.drug_confidence(x)
    
    syr_info=syr_peaks_check(data_of_report['unknown'],data_of_report['known_peaks'],10)
    
    # print(data_of_report['unknown'],'old unknown')
  
    local_db = drug_report.db
    ref = local_db['raw'] #xcal
    xnorm = peak_normalization(x)
     
    score_report = q_scoring(xnorm,ref)
#     return data_of_report

    data_of_report = {**data_of_report,**score_report,**syr_info}
    
    if data_of_report['score']>local_db['pass_threshold']: data_of_report['alert'] = 'low'
    elif data_of_report['score']>0.45 and data_of_report['score']<local_db['pass_threshold']: data_of_report['alert'] = "medium"
    else: data_of_report['alert'] = 'high'
        
    if data_of_report['alert']=='high': 
        data_of_report['found_drug'] = list(map(lambda x: x.lower(), data_of_report['found_drug']))
        try:
            data_of_report['found_drug'].pop(data_of_report['found_drug'].index(str(drugname).lower()))
            data_of_report['found_drug'] = list(map(lambda x: x.capitalize(), data_of_report['found_drug']))
        except Exception as e: print(e)
        data_of_report['not_found_drug'].append(drugname)
    
    data_of_report['not_found_drug'] = list(set(data_of_report['not_found_drug']))
    data_of_report['found_drug'] = list(set(data_of_report['found_drug']))
    return data_of_report


#################################Q-Learning Validation########################
     
def find_score_val(data,drug_name):
    window=15
    new= np.zeros(len(data))
    
    if len(data)== 2401: drug_peak= [i-200 for i in database_drugs[drug_name]['peaks_list']]
    elif len(data) ==1802: drug_peak= [i-350 for i in database_drugs[drug_name]['peaks_list']]
    
    for i in drug_peak:
        new[i-window:i+window]= data[i-window:i+window]
    new= butterworth_filter(new)    
    new-= als_baseline(new)    
    new= np.where(new<0,0,new) 
    peak_found= [i for i in find_peaks(new, prominence=100,rel_height=0.7,threshold=0.7)[0]]
    peak_range= [range(i-window,i+window) for i in drug_peak]     
    peaks_in_range= []
    for peak in peak_found:
        count=0   
        for j in peak_range:
            if peak in j: 
                count=1
                break
        if count==1: peaks_in_range.append(peak)
    score= len(peaks_in_range)
    
    if len(data)== 2401:
        print("Drug Peak: ", [i+200 for i in drug_peak])
        print("Found Peaks: ", [i+200 for i in peaks_in_range])
        
    if len(data)== 1802:
        print("Drug Peak: ", [i+350 for i in drug_peak])
        print("Found Peaks: ", [i+350 for i in peaks_in_range])
    
    return score

def scan_validation(data_to_save,drug_name):
    
    """
    data_to_save: saved json data
    drug_name: Name of the Drug to be validated
    """
    # gold_sign= database_drugs[drug_name]['gold_sign']
    data= butterworth_filter(data_to_save['xcal'])
    data-= als_baseline(data)
    score= find_score_val(data, drug_name)
    
    if score>=1: return True
    else: return False
         
###################################################################################
# DLearning Drug validation before adding
def feed_dlearning(x):
    global dlearning_file
    database = json.load(open(dlearning_file))
    x = str(x).capitalize()
    if x not in database['dlearning_drugs']:
        database['dlearning_drugs'].append(x)
        with open(dlearning_file,'w') as f:
            json.dump(database,f)
            print(f"Added {x} to Database")
            return True
    else:
        print("Drug is already present")
        return False    

################################## Comparision of Spectra #######################################
def to_series(x):
    if len(x)==2401: return pd.Series(x,index=np.arange(200,2601))
    elif len(x)==1802: return pd.Series(x,index=np.arange(350,2152))
    else: return pd.Series(x,index=np.arange(0,len(x)))
class Compare_Spectra:
    def __init__(self):
        pass
    def sec_ord(self,x,k=2):
        x =  np.array(x)

        for _ in range(k):
            x = np.gradient(x)
        #x =  butterworth_filter(x)
        return pd.Series((x))
    
    def second_order(self,x,k=2):
        x =  np.array(x)
        if len(x)==2401: wave = np.arange(200,2601)
        if len(x)==1802: wave = np.arange(350,2152)
        for _ in range(k):
            x = np.gradient(x,wave)
        #x =  butterworth_filter(x)
        return pd.Series((x)[200:1400])

    def compare_preprocess(self,x,y):
        x1 = preprocess(x,idx=(400,1700))
        y1 = preprocess(y,idx=(400,1700))
        #Alt = round(round(Alert().test(x1.values,y1.values)[0],3)*100,2)
        Alt = round(pearsonr(x1.values,y1.values)[0] * 100,3)
        print("Preprocess ",Alt)
        return Alt

    def compare_peaks(self,x,y):
        x1 = preprocess(x,idx=(400,1700),smoothing=True).values
        y1 = preprocess(y,idx=(400,1700),smoothing=True).values
        #x1 = norm(x)[200:1500]
        #y1 =  norm(y)[200:1500]
        #x1 =  savgol_filter(x1,51,11)
        #y1 = savgol_filter(y1,51,11)

        xp =  find_peaks(x1,height=0.05,width=3)
        yp = find_peaks(y1,height=0.05,width=3)

        c=0

        for i,j in find_all_near_peaks(xp[0],yp[0],multi=True,strict_mode=True).items():
            if len(j)>0 and abs(i-j[0])<=4: c+=1

        score =  round(c/max(len(xp[0]),len(yp[0])) * 100, 2)
        print("Score of peaks ",score)
        return score

    def compare_second_order(self,x,y):
        x1 =  self.second_order(x)
        y1 =  self.second_order(y)
        Alt = round(pearsonr(x1.values,y1.values)[0]*100,3)
        print("Second order ",Alt)
        return Alt

    def compare_flourense(self,x,y):
        x =  to_series(x).loc[400:1700]
        y = to_series(y).loc[400:1700]

        #smoothness_param,max_iters,conv_thresh=5,10,0.001 #params for testsing

        #x1 =  airpls_baseline(x,smoothness_param=smoothness_param,max_iters=max_iters,conv_thresh=conv_thresh)
        #y1 = airpls_baseline(y,smoothness_param=smoothness_param,max_iters=max_iters,conv_thresh=conv_thresh)
        #x1 =  airpls_baseline(x)
        #y1 = airpls_baseline(y)
        x1 = als_baseline(x)
        y1 = als_baseline(y)
     
        x1 =  self.sec_ord(x1)
        y1 =  self.sec_ord(y1)

        Alt = round(Alert().test(x1.values,y1.values)[0]*100,3)
        print("Compare Flu old ",Alt)
        Alt = round(pearsonr(x1,y1)[0]*100,3)
        print("Compare Flu ",Alt)
        return Alt

    def compute_area_percentage(self,a,b):
        a,b = to_series(a),to_series(b)
        x = np.trapz(pd.Series(preprocess(a.loc[400:1700])-preprocess(b.loc[400:1700])).abs())
        print('Area ' ,round(x,3))
        x = round(np.interp(x,[0,20,40,60,80,100,400,500,700,1000,1300],[100, 90, 80, 70, 60, 50, 30, 20, 10, 5, 0]),3)
        print("Area Percent ",x)
        return x

    def Avg_Compare(self,a,b):
        d =  {"structure_score":self.compare_preprocess(a,b),
              "second_order_score":self.compare_second_order(a,b),
              "flourense_score":self.compare_flourense(a,b),
              "peaks_score":self.compare_peaks(a,b),
              "area_score":self.compute_area_percentage(a,b)
             }
        d['overall_score'] = np.average(list(d.values()))
        return d
