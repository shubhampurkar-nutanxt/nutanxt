from .oak_drug_database import Structure_Table,Identification_info,Quantification_info,Solvents_info,Api_info, Syringe_info,ID2Database
import json
import pandas as pd
import os

ETL_VER =  1.4 #florida # syr_info
database = {}
database_drugs = {}
Q_DRUG,IDENTIFY_DRUG,NARCO_DRUG,API_DRUG,NON_NARCO_DRUG=[],[],[],[],[]

id_sample = os.path.dirname(os.path.realpath(__file__))+'/identification.csv'
if os.path.exists(id_sample): IDENTIFY_DRUG =  pd.read_csv(id_sample,skip_blank_lines=True,skipinitialspace=True)['drugname'].tolist()


def database_etl(dbpath):    
    s_df = Structure_Table.fetch_data(dbpath)
    i_info = Identification_info.fetch_data(dbpath)
    a_info = Api_info.fetch_data(dbpath)
    q_info =  Quantification_info.fetch_data(dbpath)
    solv_info =   Solvents_info.fetch_data(dbpath)
    syr_info = Syringe_info.fetch_data(dbpath)
    ALL_DRUGS = sorted(i_info['drugname'].tolist())

    #Filter to get only required samples
    i_info = i_info[i_info['drugname'].isin(ALL_DRUGS)]

    """def expand_based_on_solvent(df):
        expanded_data = pd.DataFrame(columns=df.columns)
        for i, row in df.iterrows():
            solvents = row["solvents"]
            for solvent in solvents:
                new_row = row.copy()
                new_row["solvents"] = solvent
                expanded_data = expanded_data.append(new_row, ignore_index=True)
        return expanded_data"""
    
    def expand_based_on_solvent(df):
        expanded_data = []
        for i, row in df.iterrows():
            solvents = row["solvents"]
            for solvent in solvents:
                new_row = row.copy()
                new_row["solvents"] = solvent
                expanded_data.append(new_row)
        expanded_data = pd.DataFrame(expanded_data)
        return expanded_data

    def peaks_generator(df, solvents_df, size=10):
        drugname = df['drugname']
        solvents  = [df['solvents']] if isinstance(df['solvents'],str) else df['solvents']
        if 'water' not in solvents: solvents.append('water')
        gen_peaks = {}
        peaks = eval(df['peaks_list']) if isinstance(df['peaks_list'],str) else df['peaks_list']

        def peaks_formatter(drugname,peak,size=10) -> dict:
                d= {}
                d[str(drugname).capitalize()+'_'+str(peak)] = [peak-size,peak+size]    
                return d

        for i in peaks:
            gen_peaks = {**gen_peaks,**peaks_formatter(drugname,i,size)}

        for sol in solvents:
            info = solvents_df[solvents_df['name'].str.lower()==str(sol).lower()]
            peaks_info =  eval(info['peaks'].values[0]) if isinstance(info['peaks'].values[0],str) else info['peaks'].values[0]
            window_size = info['window_size'].values[0]
            name = info['name'].values[0]
            for p in peaks_info: gen_peaks = {**gen_peaks,**peaks_formatter(name,p,window_size)}

        return gen_peaks
    
    #QDRUG
    QDRUG = q_info[q_info['verified'].str.lower()=='yes']['drugname'].values.tolist()
    #NARCO
    NARCO = i_info[i_info['narco'].str.lower()=='yes']['drugname'].values.tolist()
    #NON_NARCO
    NON_NARCO = [i for i in IDENTIFY_DRUG if i not in NARCO]
    #API_DRUGS
    API_DRUG = a_info.drugname.to_list()
    # SYRINGE
    SYRNAME = syr_info['name'].tolist()    
    
    # Creating UNAME for s_df
    s_df['uname']=s_df.apply(lambda x: x['drugname']+'_'+'_'.join(x['solvents']) if len(x['solvents'])>0 else x['drugname'],axis=1)
    
    
    # Mapper of columns to table in q_info 
    q_info.rename(columns={"quantification_peak":"critical_peak","peaks":"peaks_list"},inplace=True)
    q_info['confidence'] =  q_info['pass_threshold']
    
    # Expand Based Solvents list 
    q_info = expand_based_on_solvent(q_info)
    
    #peaks_creation important
    q_info['peaks'] = q_info.apply(lambda x: peaks_generator(x,solv_info),axis=1)
    
    # CREATING UNAME for q_info
    q_info['uname']=q_info.apply(lambda x: x.drugname+'_Water'+"_"+str(x.solvents).capitalize() if str(x.solvents).lower()!='water' else x.drugname+'_Water' ,axis=1)
    
    
    # Mapper of columns to table in i_info
    i_info.rename(columns={"peaks":"peaks_list"},inplace=True)
    
    # Mapper of columns to table in a_info
    a_info.rename(columns={"peaks":"peaks_list"},inplace=True)

    # Mapper of columns to table in syr_info
    syr_info.rename(columns={"peaks":"peaks_list"},inplace=True)

    # Q_info Structure
    q_info['xcal'] = q_info['uname'].apply(lambda x: s_df[s_df['uname']==x]['xcal'].values).apply(lambda y: y[0] if len(y)>0 else y)
    q_info['ycal'] = q_info['uname'].apply(lambda x: s_df[s_df['uname']==x]['ycal'].values).apply(lambda y: y[0] if len(y)>0 else y)
    q_info['raw'] = q_info['xcal'] # raw referred as xcal
    
    # I_info Structure
    i_info['xcal'] = i_info['drugname'].apply(lambda x: s_df[s_df['uname']==x]['xcal'].values).apply(lambda y: y[0] if len(y)>0 else y)
    i_info['ycal'] = i_info['drugname'].apply(lambda x: s_df[s_df['uname']==x]['ycal'].values).apply(lambda y: y[0] if len(y)>0 else y)
    i_info['raw'] = i_info['xcal'] # raw referred as xcal

    # a_info Structure
    a_info['xcal'] = a_info['drugname'].apply(lambda x: s_df[s_df['drugname'].str.lower()==x.lower()]['xcal'].values).apply(lambda y: y[0] if len(y)>0 else y)
    a_info['ycal'] = a_info['drugname'].apply(lambda x: s_df[s_df['drugname'].str.lower()==x.lower()]['ycal'].values).apply(lambda y: y[0] if len(y)>0 else y)
    a_info['raw'] = a_info['xcal'] # raw referred as xcal

    # syr_info Structure
    syr_info['xcal'] = syr_info['name'].apply(lambda x: syr_info[syr_info['name'].str.lower()==x.lower()]['xcal'].values).apply(lambda y: y[0] if len(y)>0 else y)
    syr_info['ycal'] = syr_info['name'].apply(lambda x: syr_info[syr_info['name'].str.lower()==x.lower()]['ycal'].values).apply(lambda y: y[0] if len(y)>0 else y)
    syr_info['raw'] = syr_info['xcal'] 
    # JSON_Converter
    data = {}
    for i in q_info.uname:
        dk = q_info[q_info['uname']==i]
        data = {**data,**json.loads(dk.set_index('uname').T.to_json())}
    for i in i_info.drugname:
        dk =  i_info[i_info['drugname']==i]
        data = {**data,**json.loads(dk.set_index('drugname').T.to_json())}
    for i in a_info.drugname:
        dk =  a_info[a_info['drugname']==i]
        data = {**data,**json.loads(dk.set_index('drugname').T.to_json())}
    for i in syr_info.name:
        dk =  syr_info[syr_info['name']==i]
        data = {**data,**json.loads(dk.set_index('name').T.to_json())}
        
    return {"database_drugs":data,"q_drug":QDRUG,"i_drug":IDENTIFY_DRUG,"narco_drug":NARCO,"non_narco_drug":NON_NARCO,"all_drugs":ALL_DRUGS,"a_drug":API_DRUG,'syringe_name':SYRNAME}
   
print(f"ETL Version  -> {ETL_VER}")
dbpath = os.path.dirname(os.path.realpath(__file__))+'/drug_database_db.db'
database = database_etl(dbpath)
database_drugs = database['database_drugs']
Q_DRUG = database['q_drug']
IDENTIFY_DRUG = database['i_drug'] # updated removed API_DRUG in IDENTIFY_DRUG
API_DRUG = database['a_drug']
NARCO_DRUG = database['narco_drug']
NON_NARCO_DRUG = database['non_narco_drug']
SYRNAME = database['syringe_name']   # syringe names
DRUGSLIST = database['all_drugs']
print(f"Q_DRUG = {Q_DRUG}")
print(f"IDENTIFY_DRUG = {IDENTIFY_DRUG}")
print(f'Number of IDRUG = {len(IDENTIFY_DRUG)}')
print(f"API_DRUG = {API_DRUG}")
print(f"syringe_name = {SYRNAME}")
######################################## ID2 ######################################################
ID2_DB_PATH = os.path.dirname(os.path.realpath(__file__))

ID2_DB_DATA = ID2Database(db_path=ID2_DB_PATH)