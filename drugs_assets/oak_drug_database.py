import json
import sqlite3
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def peak_normalization(df):
        if len(df)>0:
            df = np.array(df)
            if len(df)==2401:
                x = df / (df.max() - df.min())
                return x
            elif len(df)==1802:
                x = df/(df[0:1500].max() - df.min())
                return x
            else:
                x = df/(df.max()-df.min())
                return x

            
            
##############################################################################################################################

class Database_OAK():
    def __init__(self,dbpath):
        self.database = 0
        """
        dbpath(*.db): "path of sql database path"
        """
        
        if os.path.exists(dbpath) and dbpath.split('.')[-1]=='db':
            try:
                self.database = sqlite3.connect(dbpath)
            except:
                raise ValueError(f"Unable to connect to  {dbpath}")
        else:
            raise ValueError(f"Database not found at {dbpath}")
            
        self.structure_ref_type={
            "drugname":{"type":str},
            "solvents":{"type":list},
            "xcal": {"type":list,"size":2401},
            "ycal":{"type":list,"size":1802},
            "raw":{"type":list,"size":1920}
        }
        self.id_table_ref_type={
            "drugname":{"type":str},
            "peaks":{"type":list,"range":[200,2601]},
            "highest_drug_peak":{"type":list,"range":[200,2601]},
            "critical_peak":{"type":list,"range":[200,2601]},
            "peak_threshold":{"type":float,"range":[0,1]},
            "pass_threshold":{"type":float,"range":[0,1]},
            "drug_limit":{"type":float,"range":[0,100]},
            "verified":{"type":str},
            "narco":{"type":str}
        }
        
        self.quantification_table_ref_type = {
            "drugname":{"type":str},
            "peaks":{"type":list,"range":[200,2601]},
            "peak_threshold":{"type":float,"range":[0,1]},
            "quantification_peak":{"type":list,"range":[200,2601]},
            "highest_conc":{"type":int,"range":[1,100]},
            "lowest_conc":{"type":float,"range":[0,100]},
            "solvents":{"type":list},
            "pass_threshold":{"type":float,"range":[0,1]},
            "verified":{"type":str}
        }
        
        self.pharmacy_table_ref_type = {
            "drugname":{"type":str},
            "key":{"type":str},
            "xcal_pure":{"type":list,"size":2401},
            "peaks":{"type":list,"range":[200,2601]},
            "peak_threshold":{"type":float,"range":[0,1]},
            "quantification_peak":{"type":list,"range":[200,2601]},
            "critical_peak":{"type":list,"range":[200,2601]},
            "highest_conc":{"type":int,"range":[1,100]},
            "lowest_conc":{"type":int,"range":[1,100]},
            "solvents":{"type":list},
            "pass_threshold":{"type":float,"range":[0,1]},
            "verified":{"type":str}
        }

        self.api_table_ref_type={
            "drugname":{"type":str},
            "peaks":{"type":list,"range":[200,2601]},
            "max_weight":{"type":float,"range":[0,1000]},
            "highest_drug_peak":{"type":list,"range":[200,2601]},
            "critical_peak":{"type":list,"range":[200,2601]},
            "quantification_peak":{"type":list,"range":[200,2601]},
            "min_conc":{"type":object,"range":[1,100]},
            "max_conc":{"type":object,"range":[1,100]},
            "min_conc_ycal":{"type":list,"size":1802},
            "max_conc_ycal":{"type":list,"size":1802},
            "q_threshold":{"type":list,"range":[0,100]},
            "peak_threshold":{"type":float,"range":[0,1]},
            "pass_threshold":{"type":float,"range":[0,1]},
            "drug_limit":{"type":float,"range":[0,100]}
        }
        
        self.syringe_info_ref_type={
            "name":{"type":str},
            "peaks":{"type":list,"range":[200,2601]},
            "xcal": {"type":list,"size":2401},
            "ycal":{"type":list,"size":1802},
            "raw":{"type":list,"size":1920}
        }

        self.solvents_table_ref_type = {
            "name":{"type":str},
            "peaks":{"type":list,"range":[200,2601]},
            "window_size":{"type":int}  
        }
        
    def show_tables(self):
        query = """SELECT name FROM sqlite_master WHERE type='table';"""
        if self.database!=0 and isinstance(self.database,sqlite3.Connection):
            cur = self.database.cursor()
            tables = cur.execute(query).fetchall()
            tables = "\n".join([i[0] for i in tables])
            print("Tables: ")
            print(tables)
        else:
            print("Connection is closed ")
    
    def close_database(self):
        if self.database!=0 and isinstance(self.database,sqlite3.Connection):
            self.database.close()
            print("Database Connection_Closed")
            return True
        else:
            return False
  
##############################################################################################################################
class Structure_Table(Database_OAK):
    """
    Used to perform insert data to structure table of Oak_Database
    Attributes:
        dbpath(*.db): "path of sql database path"
        drugname(str): "drugname of new sample"
        solvents(list): "list of solvents in that struture default ['Water']"
        file(str): "path of json record with xcal,ycal,raw" or file(dict): "with xcal,yal,raw as keys and values of spectrum of xcal,ycal,raw respectively"
        structure_ref_type: reference for respective attribute and there type and size
        kwargs looks for xcal ycal raw 
    methods:
        view_instance used to plot data of file provide at creating an instance
        insert_to_db used to insert data to table 
        fetch_data returns the rows present in table
        
        CREATE TABLE "structure" (
        "drugname"	TEXT NOT NULL,
        "solvents"	BLOB NOT NULL DEFAULT '[''Water'']',
        "xcal"	BLOB,
        "ycal"	BLOB,
        "raw"	BLOB,
        PRIMARY KEY("drugname","solvents")
    )
    """
    def __init__(self,dbpath,drugname:str,file=None,solvents=[],apply_peak_normalization=True,**kwargs):
        """
        dbpath(*.db): "path of sql database path"
        drugname(str): "drugname of new sample"
        file(str): "path of json record with xcal,ycal,raw" or file(dict): "with xcal,yal,raw as keys and values of spectrum of xcal,ycal,raw respectively"
        kwargs looks for xcal ycal raw     
        """
        super().__init__(dbpath)
        self.table_name = 'structure'
        self.dbpath = dbpath
        self.drugname = str(drugname).capitalize()
        
        if isinstance(file,str) and file.split('.')[-1]=='json':
            data = json.load(open(file))
            if apply_peak_normalization:
                self.xcal = peak_normalization(data['xcal'])
                self.ycal = peak_normalization(data['ycal'])
                self.raw = peak_normalization(data['raw'])
            else:
                self.xcal = data['xcal']
                self.ycal = data['ycal']
                self.raw = data['raw']     
        elif isinstance(file,dict):
            assert(sorted(file.keys()) == ['raw','xcal','ycal'])
            if apply_peak_normalization:
                self.xcal = peak_normalization(data['xcal'])
                self.ycal = peak_normalization(data['ycal'])
                self.raw = peak_normalization(data['raw'])
            else:
                self.xcal = data['xcal']
                self.ycal = data['ycal']
                self.raw = data['raw'] 
        elif file is None:
            if apply_peak_normalization:
                self.xcal = peak_normalization(kwargs['xcal'])
                self.ycal = peak_normalization(kwargs['ycal'])
                self.raw = peak_normalization(kwargs['raw'])
            else:
                self.xcal = kwargs['xcal']
                self.ycal = kwargs['ycal']
                self.raw = kwargs['raw']
            
        if isinstance(solvents,self.structure_ref_type['solvents']['type']):
            self.solvents =  solvents
        else: raise ValueError("Solvents must be in list eg ['Water','Dextrose']")
            
    def __repr__(self):
        return f"Structure Table\ninstance = {self.drugname},\nlength of xcal = {len(self.xcal)},\nlength of ycal = {len(self.ycal)},\nlength of raw = {len(self.raw)}\nSolvents = {self.solvents}"
    
    def __str__(self):
        return f"Structure Table\ninstance = {self.drugname},\nlength of xcal = {len(self.xcal)},\nlength of ycal = {len(self.ycal)},\nlength of raw = {len(self.raw)}\nSolvents = {self.solvents}"       
    
 
    def view_instance(self):
        """
        this method is used to plot xcal,ycal,raw of the instance
        """
        fig,ax = plt.subplots(ncols=3,nrows=1,figsize=(12,4))
        for num,level in enumerate(zip(['xcal','ycal','raw'],[self.xcal,self.ycal,self.raw])):

            ax[num].plot(level[1])
            ax[num].set_title(level[0])
        plt.show()
              
    def verify_insertion_type(self,cun):
        row = cun.execute(f"SELECT * FROM {self.table_name} WHERE drugname = '{self.drugname}'").fetchone()#[0]
        keys =  self.structure_ref_type.keys()
        for r in zip(row,keys):
            if r[1]!='drugname':
                if (isinstance(eval(r[0]),self.structure_ref_type[r[1]]['type']) and len(eval(r[0]))==self.structure_ref_type[r[1]]['size']) ==False:
                    raise ValueError("Not matching the requirement")    
            else:
                if isinstance(r[0],self.structure_ref_type[r[1]]['type'])==False:
                    raise ValueError("Not matching the requirement") 
                
            
    def insert_to_db(self,view=True):
        """
        Inserted the data to table for current instance, verify the instance by using view_instance
        view: used to view the data of instance before inserting to table
        """
        cun = self.database.cursor()
        if view:
            self.view_instance()
        try:
            print(f"Trying to insert {self.drugname} to {self.table_name} table")
            cun.execute(f"INSERT INTO {self.table_name} (drugname,solvents,xcal,ycal,raw) VALUES(?,?,?,?,?)",(self.drugname,str(list(self.solvents)),str(list(self.xcal)),str(list(self.ycal)),str(list(self.raw))));
            print("\nVerification of type is in process.....\n")
            #self.verify_insertion_type(cun)
            print("Verfied Successfully")
            inp = input("\nEnter yes or y to commit on database...\n ")
            if str(inp).lower()=='yes' or str(inp).lower() == 'y':
                self.database.commit()
                print(f"{self.drugname} is inserted to {self.table_name} table")
            else:
                print("\nAborted the commit\n")
        except Exception as e:
            self.database.rollback()
            print("Failed to insert {self.drugname} to {self.table_name} by {e}")
        finally:
            super().close_database()
            
    @staticmethod
    def fetch_data(dbpath):
        """
        fetch the entire data from table structure of Oak_database
        """
        import pandas as pd
        database = Database_OAK(dbpath)
        df =  pd.read_sql("select * from structure",database.database)
        for col in df.columns:
            if database.structure_ref_type[col]['type']==list:
                df[col]= df[col].apply(lambda x: eval(x))     
        database.close_database()
        return df
    
    @staticmethod
    def update_row(dbpath,drugname,column,value):
        """
        Update the value for specific column in structure columns
        """
        database = Database_OAK(dbpath)
        cur = database.database.cursor()
        
        try:
            print("Verifying process started...")
            print("Verified beforing updating...")
            cur.execute(f'''UPDATE structure SET {column} = '{value}' WHERE drugname='{drugname}';''')
            #verifying_Query(cur,drugname,database)
            print("Verified after updated")
            inp = input("\nEnter yes or y to commit on database...\n ")
            if str(inp).lower()=='yes' or str(inp).lower() == 'y':
                                database.database.commit()
                                print(f"{column} updated for {drugname} by {value}")
            else:
                                print("\nAborted the commit\n")
                    
            
        except Exception as e:
            database.database.rollback()
            print(f"Failed to update {column} for {drugname} by {e}")
        finally:
            database.close_database()
            del cur,database

    @staticmethod
    def delete_row(dbpath, drugname):
        """
        Delete a row with the given drugname from the structure table.
        """
        database = Database_OAK(dbpath)
        cur = database.database.cursor()

        try:
            print("Deleting row from structure...")
            cur.execute(f"DELETE FROM structure WHERE drugname = '{drugname}';")
            # verifying_Query(cur,drugname,database)
            print("Verified after deleting")
            inp = input("\nEnter yes or y to commit on database...\n ")
            if str(inp).lower() == 'yes' or str(inp).lower() == 'y':
                database.database.commit()
                print(f"{drugname} row deleted from structure table.")
            else:
                print("\nAborted the commit\n")

        except Exception as e:
            database.database.rollback()
            print(f"Failed to delete {drugname} row from structure table by {e}")
        finally:
            database.close_database()
            del cur, database

 
##############################################################################################################################
class Quantification_info(Database_OAK):
    """
    Used to perform operation of quantification_info of OAK Database
    Attributes:
        drugname(str): "drugname of new sample"
        peaks(list): "list of drug peaks eg [1233,1344,1634]"
        peak_threshold(float): default(0.04), peak identification limit for drug that is promience for peak finding in quantification process
        quantification_peak(list): default [], quantification peak used to quantify in drug
        highest_conc(int): highest concentration drug limit
        lowest_conc(int): lowest concentartion drug limit
        solvents(list): list of solvents that specific drug as been seen
        pass_threshold(float): default(0.7), pass_threshold for drug that can be consider as pass_threshold as pass that is 0.7 for structure similiarity q_identification
        verified("string"): default (No), if yes the drug is authenticated for identification process
        quantification_table_ref_type: reference for respective attribute and there type and there range
    
    STATEMENT:
        CREATE TABLE "quantification_info" (
        "drugname"	TEXT NOT NULL,
        "peaks"	BLOB,
        "peak_threshold"	NUMERIC,
        "quantification_peak"	BLOB,
        "highest_conc"	NUMERIC,
        "lowest_conc"	NUMERIC,
        "solvents"	BLOB,
        "pass_threshold"	NUMERIC DEFAULT 0.7,
        "verified"	TEXT,
        PRIMARY KEY("drugname")
    )
    """
    
    def __init__(self,dbpath,
                drugname:str,
                 peaks: list=[],
                 peak_threshold: float=0.04,
                 quantification_peak: list=[],
                 highest_conc: int = 20,
                 lowest_conc: float =  1,
                 solvents: list = ['water'],
                 pass_threshold: float = 0.7,
                  verified: str = 'No'
                ):
        """
            dbpath(*.db): "path of sql database path"
            drugname(str): "drugname of new sample"
            peaks(list): "list of drug peaks eg [1233,1344,1634]"
            peak_threshold(float): default(0.04), peak identification limit for drug that is promience for peak finding in q_identification process
            highest_conc(int): default 20, highest drug peak of drug sample
            lowest_conc(int): default 1, lowest drug peak of drug sample
            solvents(list): default [], solvents in sample
            pass_threshold(float): default(0.7), pass_threshold for drug that can be consider as pass_threshold as pass that is 0.7
            verified("string"): default (No), if yes the drug is authenticated for identification process
        """
        
        super().__init__(dbpath)  
        self.table_name = "quantification_info"
        self.dbpath = dbpath
        self.drugname = str(drugname).capitalize()
        
        if isinstance(peaks,self.quantification_table_ref_type['peaks']['type']) and min(peaks)>self.quantification_table_ref_type['peaks']['range'][0] and max(peaks)<self.quantification_table_ref_type['peaks']['range'][1]: self.peaks = peaks
        else: raise ValueError("peaks must be list eg: [1002,1408,1038] and must be in range 200 and 2601")
            
        if isinstance(quantification_peak,self.quantification_table_ref_type['quantification_peak']['type']):
            if len(quantification_peak)!=0 and min(quantification_peak)>self.quantification_table_ref_type['quantification_peak']['range'][0] and max(quantification_peak)<self.quantification_table_ref_type['quantification_peak']['range'][1]: self.quantification_peak = quantification_peak
            else: self.quantification_peak = quantification_peak
        else:
            raise ValueError("quantification peak must be list and must be in range 200 and 2601")
            
        if isinstance(highest_conc,self.quantification_table_ref_type['highest_conc']['type']) and highest_conc>0: self.highest_conc = highest_conc
        else: raise ValueError("highest_conc must be int and must be more than 0")
        
        if isinstance(lowest_conc,self.quantification_table_ref_type['lowest_conc']['type']) and lowest_conc>0:self.lowest_conc = lowest_conc
        else: raise ValueError("lowest_conc must be float or int and must be more than 0")
        
        if isinstance(solvents,self.quantification_table_ref_type['solvents']['type']): self.solvents=solvents
        else: raise ValueError("solvents must be in list")
            
        if (isinstance(peak_threshold,self.quantification_table_ref_type['peak_threshold']['type']) or isinstance(peak_threshold,int)) and peak_threshold>=self.quantification_table_ref_type['peak_threshold']['range'][0] and peak_threshold<=self.quantification_table_ref_type['peak_threshold']['range'][1]:self.peak_threshold = peak_threshold
        else: raise ValueError("peak_threshold must be float eg: 0.04 and must be in range [0,1]")
        
        if (isinstance(pass_threshold,self.quantification_table_ref_type['pass_threshold']['type']) or isinstance(pass_threshold,int)) and pass_threshold>=self.quantification_table_ref_type['pass_threshold']['range'][0] and pass_threshold<=self.quantification_table_ref_type['pass_threshold']['range'][1] : self.pass_threshold = pass_threshold 
        else: raise ValueError("pass_threshold must be float eg: 0.7 and must be in range [0,1]")
    
        self.verified = str(verified).capitalize() if str(verified).lower() in ['yes','no'] else "No"
        
    def __repr__(self):
        return f"Quantification_Info Table \ndrugname = {self.drugname}\npeaks = {self.peaks}\nhighest_conc = {self.highest_conc}\nlowest_conc = {self.lowest_conc}\nsolvents = {self.solvents}\npeak_threshold = {self.peak_threshold}\npass_threshold = {self.pass_threshold}\nverified = {self.verified}"
    
    def __str__(self):
        return f"Quantification_Info Table \ndrugname = {self.drugname}\npeaks = {self.peaks}\nhighest_conc = {self.highest_conc}\nlowest_conc = {self.lowest_conc}\nsolvents = {self.solvents}\npeak_threshold = {self.peak_threshold}\npass_threshold = {self.pass_threshold}\nverified = {self.verified}"
    
    
    def view_instance(self):
        print(f"Quantification_Info Table \ndrugname = {self.drugname}\npeaks = {self.peaks}\nhighest_conc = {self.highest_conc}\nlowest_conc = {self.lowest_conc}\nsolvents = {self.solvents}\npeak_threshold = {self.peak_threshold}\npass_threshold = {self.pass_threshold}\nverified = {self.verified}")
    
    
    
    def insert_to_db(self,view=True):
        """
        Inserted the data to table for current instance, verify the instance by using view_instance
        view: used to view the data of instance before inserting to table
        """
        cun = self.database.cursor()
        if view: self.view_instance()
        try:
            print(f"Trying to insert {self.drugname} to {self.table_name} table")
            cun.execute(f"INSERT INTO {self.table_name} (drugname,peaks,peak_threshold,quantification_peak,highest_conc,lowest_conc,solvents,pass_threshold,verified) VALUES(?,?,?,?,?,?,?,?,?)",(self.drugname,str(list(self.peaks)),self.peak_threshold,str(list(self.quantification_peak)),self.highest_conc,self.lowest_conc,str(self.solvents),self.pass_threshold,self.verified));
            print("\nVerification of type is in process.....\n")
            #self.verify_insertion_type(cun)
            print("Verfied Successfully")
            inp = input("\nEnter yes or y to commit on database...\n ")
            if str(inp).lower()=='yes' or str(inp).lower() == 'y':
                self.database.commit()
                print(f"{self.drugname} is inserted to {self.table_name} table")
            else:
                print("\nAborted the commit\n")
        except Exception as e:
            self.database.rollback()
            print(f"Failed to insert {self.drugname} to {self.table_name} by {e}")
        finally:
            super().close_database()

    @staticmethod
    def update_row(dbpath,drugname,column,value):
        """
        Update the value for specific column in quantification_info columns
        """
        database = Database_OAK(dbpath)
        cur = database.database.cursor()
        
        try:
            print("Verifying process started...")
            print("Verified beforing updating...")
            cur.execute(f'''UPDATE quantification_info SET {column} = '{value}' WHERE drugname='{drugname}';''')
            #verifying_Query(cur,drugname,database)
            print("Verified after updated")
            inp = input("\nEnter yes or y to commit on database...\n ")
            if str(inp).lower()=='yes' or str(inp).lower() == 'y':
                                database.database.commit()
                                print(f"{column} updated for {drugname} by {value}")
            else:
                                print("\nAborted the commit\n")
                    
            
        except Exception as e:
            database.database.rollback()
            print(f"Failed to update {column} for {drugname} by {e}")
        finally:
            database.close_database()
            del cur,database

    @staticmethod
    def delete_row(dbpath, drugname):
        """
        Delete a row with the given drugname from the quantification_info table.
        """
        database = Database_OAK(dbpath)
        cur = database.database.cursor()

        try:
            print("Deleting row from quantification_info...")
            cur.execute(f"DELETE FROM quantification_info WHERE drugname = '{drugname}';")
            # verifying_Query(cur,drugname,database)
            print("Verified after deleting")
            inp = input("\nEnter yes or y to commit on database...\n ")
            if str(inp).lower() == 'yes' or str(inp).lower() == 'y':
                database.database.commit()
                print(f"{drugname} row deleted from quantification_info table.")
            else:
                print("\nAborted the commit\n")

        except Exception as e:
            database.database.rollback()
            print(f"Failed to delete {drugname} row from quantification_info table by {e}")
        finally:
            database.close_database()
            del cur, database
            
    @staticmethod
    def fetch_data(dbpath):
        """
        fetch the entire data from table quantification_info of Oak_database
        """
        import pandas as pd
        database = Database_OAK(dbpath)
        dk =  pd.read_sql("select * from quantification_info",database.database)
        for col in dk.columns:
            if database.quantification_table_ref_type[col]['type']==list:
                dk[col]=dk[col].apply(lambda x: eval(x))
        database.close_database()
        return dk
        
        
##############################################################################################################################            
class Identification_info(Database_OAK):
    """
    Used to perform operation of identification_info of OAK Database
    Attributes:
        drugname(str): "drugname of new sample"
        peaks(list): "list of drug peaks eg [1233,1344,1634]"
        highest_durg_peak(list): default [], highest drug peak of drug in 100%(pure) drug sample
        critical_peak(list): default [], critical peak to be identifed in drug
        peak_threshold(float): default(0.04), peak identification limit for drug that is promience for peak finding in identification process
        pass_threshold(float): default(0.7), pass_threshold for drug that can be consider as pass_threshold as pass that is 0.7
        drug_limit(float): default (25), minimum identification limit of drug in mixture that is 25% of durg and 75% of cutting substance
        verified("string"): default (No), if yes the drug is authenticated for identification process
         narco("string"): default(No), if yes drug is narco drug 
        id_table_ref_type: reference for respective attribute and there type and there range
    
    
    
    STATEMENT:
        CREATE TABLE "identification_info" (
        "drugname	"	TEXT NOT NULL,
        "peaks"	BLOB,
        "highest_drug_peak"	BLOB,
        "critical_peak"	BLOB,
        "peaks_threshold"	NUMERIC,
        "pass_threshold"	NUMERIC,
        "drug_limit"	INTEGER,
        "verified"	TEXT
        "narco"->TEXT
    )
        
    """
    
    def __init__(self,dbpath,
                 drugname:str,
                 peaks:list,
                 highest_drug_peak:list = [],
                 critical_peak:list=[],
                 peak_threshold:float=0.04,
                 pass_threshold:float=0.7,
                 drug_limit:float=25,
                 verified:str = "no",
                 narco:str = "no"
                ):
        """
        dbpath(*.db): "path of sql database path"
        drugname(str): "drugname of new sample"
        peaks(list): "list of drug peaks eg [1233,1344,1634]"
        highest_durg_peak(list): default [], highest drug peak of drug in 100%(pure) drug sample
        critical_peak(list): default [], critical peak to be identifed in drug
        peak_threshold(float): default(0.04), peak identification limit for drug that is promience for peak finding in identification process
        pass_threshold(float): default(0.7), pass_threshold for drug that can be consider as pass_threshold as pass that is 0.7
        drug_limit(float): default (25), minimum identification limit of drug in mixture that is 25% of durg and 75% of cutting substance
        verified("string"): default (No), if yes the drug is authenticated for identification process\
        narco("string"): default (no), if narco drug yes else no
        """
        super().__init__(dbpath)
        
        self.table_name = "identification_info"
        self.dbpath = dbpath
        self.drugname = str(drugname).capitalize()
        
        if isinstance(peaks,self.id_table_ref_type['peaks']['type']) and min(peaks)>self.id_table_ref_type['peaks']['range'][0] and max(peaks)<self.id_table_ref_type['peaks']['range'][1]: self.peaks = peaks
        else: raise ValueError("peaks must be list eg: [1002,1408,1038] and must be in range 200 and 2601")
        
        if isinstance(highest_drug_peak,self.id_table_ref_type['highest_drug_peak']['type']) or (min(highest_drug_peak)>self.id_table_ref_type['highest_drug_peak']['range'][0] and max(highest_drug_peak)<self.id_table_ref_type['highest_drug_peak']['range'][1]): self.highest_drug_peak = highest_drug_peak
        else:raise ValueError("highest drug peak must be list eg: [1003] and must be in range 200 and 2601")
        
        if isinstance(critical_peak,self.id_table_ref_type['critical_peak']['type']) or (min(critical_peak)>self.id_table_ref_type['critical_peak']['range'][0] and max(critical_peak)<self.id_table_ref_type['critical_peak']['range'][1]) : self.critical_peak = critical_peak      
        else: raise ValueError("critical peak must be list eg: [1002] and must be in range 200 and 2601")
        
        if (isinstance(peak_threshold,self.id_table_ref_type['peak_threshold']['type']) or isinstance(peak_threshold,int)) and peak_threshold>=self.id_table_ref_type['peak_threshold']['range'][0] and peak_threshold<=self.id_table_ref_type['peak_threshold']['range'][1]:self.peak_threshold = peak_threshold
        else: raise ValueError("peak_threshold must be float eg: 0.04 and must be in range [0,1]")
        
        if (isinstance(pass_threshold,self.id_table_ref_type['pass_threshold']['type']) or isinstance(pass_threshold,int)) and pass_threshold>=self.id_table_ref_type['pass_threshold']['range'][0] and pass_threshold<=self.id_table_ref_type['pass_threshold']['range'][1] : self.pass_threshold = pass_threshold 
        else: raise ValueError("pass_threshold must be float eg: 0.7 and must be in range [0,1]")
        
        if (isinstance(drug_limit,self.id_table_ref_type['drug_limit']['type']) or isinstance(drug_limit,int)) and drug_limit>=self.id_table_ref_type['drug_limit']['range'][0] and drug_limit<=self.id_table_ref_type['drug_limit']['range'][1] : self.drug_limit = drug_limit
        else: raise ValueError("drug_limit must be float eg: 25 and must be in range [0,100]")
            
        self.verified = str(verified).lower() if str(verified).lower() in ['yes','no'] else "no"
        self.narco = str(narco).lower() if str(narco).lower() in ['yes','no'] else "no"
        
    def __repr__(self):
        return f"Identification_Info Table \ndrugname = {self.drugname}\npeaks = {self.peaks}\nhighest_drug_peak = {self.highest_drug_peak}\ncritical_peak = {self.critical_peak}\npeak_threshold = {self.peak_threshold}\npass_threshold = {self.pass_threshold}\nverified = {self.verified}"
    
    def __str__(self):
        return f"Identification_Info Table \ndrugname = {self.drugname}\npeaks = {self.peaks}\nhighest_drug_peak = {self.highest_drug_peak}\ncritical_peak = {self.critical_peak}\npeak_threshold = {self.peak_threshold}\npass_threshold = {self.pass_threshold}\nverified = {self.verified}"
    
    
    def view_instance(self):
        print(f"Identification_Info Table \ndrugname = {self.drugname}\npeaks = {self.peaks}\nhighest_drug_peak = {self.highest_drug_peak}\ncritical_peak = {self.critical_peak}\npeak_threshold = {self.peak_threshold}\npass_threshold = {self.pass_threshold}\nverified = {self.verified}")
    
    def verify_insertion_type(self,cun):
        row = cun.execute(f"SELECT * FROM {self.table_name} WHERE drugname = '{self.drugname}'").fetchone()#[0]
        keys = self.id_table_ref_type.keys()
        for num,r in enumerate(zip(row,keys)):
            if r[1] in ['highest_drug_peak',"peaks","critical_peak"]:
                if ((isinstance(eval(r[0]),self.id_table_ref_type[r[1]]['type'])) or ((min(eval(r[0]))>=self.id_table_ref_type[r[1]]['range'][0] and max(eval(r[0]))<=self.id_table_ref_type[r[1]]['range'][1])))==False:
                    raise ValueError(f"Not matching the requirement for {r[1]} expecting = {self.id_table_ref_type[r[1]]['type']}")
            
            elif r[1] in ['peak_threshold','pass_threshold','drug_limit']:
                if (isinstance(r[0],self.id_table_ref_type[r[1]]['type']) or isinstance(r[0],int)) and (r[0]>=self.id_table_ref_type[r[1]]['range'][0] and r[0]<=self.id_table_ref_type[r[1]]['range'][1])==False:
                    raise ValueError(f"Not matching the requirement for {r[1]} expecting = {self.id_table_ref_type[r[1]]['type']}")
            else:
                if (isinstance(r[0],self.id_table_ref_type[r[1]]['type']))==False:
                    raise ValueError(f"Not matching the requirement for {r[1]} expecting = {self.id_table_ref_type[r[1]]['type']}")
                    
            
    
    def insert_to_db(self,view=True):
        """
        Inserted the data to table for current instance, verify the instance by using view_instance
        view: used to view the data of instance before inserting to table
        """
        cun = self.database.cursor()
        if view: self.view_instance()
        try:
            print(f"Trying to insert {self.drugname} to {self.table_name} table")
            cun.execute(f"INSERT INTO {self.table_name} (drugname,peaks,highest_drug_peak,critical_peak,peak_threshold,pass_threshold,drug_limit,verified,narco) VALUES(?,?,?,?,?,?,?,?,?)",(self.drugname,str(list(self.peaks)),str(list(self.highest_drug_peak)),str(list(self.critical_peak)),self.peak_threshold,self.pass_threshold,self.drug_limit,self.verified,self.narco));
            print("\nVerification of type is in process.....\n")
            self.verify_insertion_type(cun)
            print("Verfied Successfully")
            inp = input("\nEnter yes or y to commit on database...\n ")
            if str(inp).lower()=='yes' or str(inp).lower() == 'y':
                self.database.commit()
                print(f"{self.drugname} is inserted to {self.table_name} table")
            else:
                print("\nAborted the commit\n")
        except Exception as e:
            self.database.rollback()
            print(f"Failed to insert {self.drugname} to {self.table_name} by {e}")
        finally:
            super().close_database()
     
    @staticmethod
    def update_row(dbpath,drugname,column,value):
        """
        Update the value for specific column in identification_info columns
        """
        database = Database_OAK(dbpath)
        cur = database.database.cursor()
        
        def verifying_Query(cur,drugname,database):
            row = cur.execute(f"SELECT * FROM identification_info WHERE drugname = '{drugname}'").fetchone()
            cols = [col[0] for col in cur.description]
            for col,data in zip(cols,row):
                if col in ['highest_drug_peak',"peaks","critical_peak"]:
                    data = eval(data)
                verifying(database,col,data)
            
            
        def verifying(database,column,value):
            if column in ['peak_threshold','pass_threshold','drug_limit']:
                if ((isinstance(value,database.id_table_ref_type[column]['type']) or isinstance(value,int)) and (value>=database.id_table_ref_type[column]['range'][0] and value<=database.id_table_ref_type[column]['range'][1]))==False:
                    raise ValueError(f"Value is not matching the reference type and range {database.id_table_ref_type[column]['range']}")
            elif column in ['highest_drug_peak',"peaks","critical_peak"]:
                if (isinstance(value,database.id_table_ref_type[column]['type'])) and (min(value)>database.id_table_ref_type[column]['range'][0] and max(value)<database.id_table_ref_type[column]['range'][1])==False:
                    raise ValueError(f"Value is not matching the reference type expected {database.id_table_ref_type[column]['type']}")
            elif column != 'drugname':
                if (isinstance(value,database.id_table_ref_type[column]['type']))==False:
                    raise ValueError(f"Value is not matching the reference type expected {database.id_table_ref_type[column]['type']}")
            elif column=='drugname':
                pass
            else: raise AttributeError(f'Column name not found in table,\ncolumns present in table = {database.id_table_ref_type.keys()}')
        
        try:
            print("Verifying process started...")
            verifying(database,column,value)
            if column in ['highest_drug_peak',"peaks","critical_peak"]:
                            value = str(value)
            print("Verified beforing updating...")
            cur.execute(f'''UPDATE identification_info SET {column} = '{value}' WHERE drugname='{drugname}';''')
            verifying_Query(cur,drugname,database)
            print("Verified after updated")
            inp = input("\nEnter yes or y to commit on database...\n ")
            if str(inp).lower()=='yes' or str(inp).lower() == 'y':
                                database.database.commit()
                                print(f"{column} updated for {drugname} by {value}")
            else:
                                print("\nAborted the commit\n")
                    
            
        except Exception as e:
            database.database.rollback()
            print(f"Failed to update {column} for {drugname} by {e}")
        finally:
            database.close_database()
            del cur,database
            
    @staticmethod
    def delete_row(dbpath, drugname):
        """
        Delete a row with the given drugname from the identification_info table.
        """
        database = Database_OAK(dbpath)
        cur = database.database.cursor()

        try:
            print("Deleting row from identification_info...")
            cur.execute(f"DELETE FROM identification_info WHERE drugname = '{drugname}';")
            # verifying_Query(cur,drugname,database)
            print("Verified after deleting")
            inp = input("\nEnter yes or y to commit on database...\n ")
            if str(inp).lower() == 'yes' or str(inp).lower() == 'y':
                database.database.commit()
                print(f"{drugname} row deleted from identification_info table.")
            else:
                print("\nAborted the commit\n")

        except Exception as e:
            database.database.rollback()
            print(f"Failed to delete {drugname} row from identification_info table by {e}")
        finally:
            database.close_database()
            del cur, database           
        
    @staticmethod
    def fetch_data(dbpath):
        """
        fetch the entire data from table identification_info of Oak_database
        """
        import pandas as pd
        database = Database_OAK(dbpath)
        df =  pd.read_sql("select * from identification_info",database.database)
        for col in df.columns:
            if database.id_table_ref_type[col]['type']==list:
                df[col]= df[col].apply(lambda x: eval(x))  
        database.close_database()
        return df
        
##############################################################################################################################
class Api_info(Database_OAK):
    """
    Used to perform operation of api_info of OAK Database
    Attributes:
        drugname(str): "drugname of new sample"
        peaks(list): "list of drug peaks eg [1233,1344,1634]"
        highest_durg_peak(list): default [], highest drug peak of drug in 100%(pure) drug sample
        critical_peak(list): default [], critical peak to be identifed in drug
        peak_threshold(float): default(0.04), peak identification limit for drug that is promience for peak finding in identification process
        pass_threshold(float): default(0.7), pass_threshold for drug that can be consider as pass_threshold as pass that is 0.7
        drug_limit(float): default (25), minimum identification limit of drug in mixture that is 25% of durg and 75% of cutting substance
        verified("string"): default (No), if yes the drug is authenticated for identification process
    
    
    
    STATEMENT:
        CREATE TABLE "api_info" (
        "drugname	"	TEXT NOT NULL,
        "max_weight" NUMERIC,
        "min_conc" NUMERIC,
        "max_conc" NUMERIC,
        "min_conc_ycal" NUMERIC,
        "max_conc_ycal" NUMERIC,
        "peaks"	BLOB,
        "highest_drug_peak"	BLOB,
        "critical_peak"	BLOB,
        "q_threshol" BLOB,
        "peaks_threshold"	NUMERIC,
        "pass_threshold"	NUMERIC,
        "drug_limit"	INTEGER,
    )
        
    """
    
    def __init__(self,dbpath,
                 drugname:str,
                 peaks:list,
                 max_weight:float,
                 highest_drug_peak:list = [],
                 critical_peak:list=[],
                 quantification_peak: list=[],
                 min_conc:float=2.5,
                 max_conc: float=20,
                 min_conc_ycal: list = [],
                 max_conc_ycal: list = [],
                 q_threshold: list = [],
                 peak_threshold:float=0.04,
                 pass_threshold:float=0.7,
                 drug_limit:float=25
                ):
        """
        dbpath(*.db): "path of sql database path"
        drugname(str): "drugname of new sample"
        peaks(list): "list of drug peaks eg [1233,1344,1634]"
        highest_durg_peak(list): default [], highest drug peak of drug in 100%(pure) drug sample
        critical_peak(list): default [], critical peak to be identifed in drug
        peak_threshold(float): default(0.04), peak identification limit for drug that is promience for peak finding in identification process
        pass_threshold(float): default(0.7), pass_threshold for drug that can be consider as pass_threshold as pass that is 0.7
        drug_limit(float): default (25), minimum identification limit of drug in mixture that is 25% of durg and 75% of cutting substance
        """
        super().__init__(dbpath)
        
        self.table_name = "api_info"
        self.dbpath = dbpath
        self.drugname = str(drugname).capitalize()
        
        if isinstance(peaks,self.api_table_ref_type['peaks']['type']) and min(peaks)>self.api_table_ref_type['peaks']['range'][0] and max(peaks)<self.api_table_ref_type['peaks']['range'][1]: self.peaks = peaks
        else: raise ValueError("peaks must be list eg: [1002,1408,1038] and must be in range 200 and 2601")
    
        if (isinstance(max_weight,self.api_table_ref_type['max_weight']['type']) or isinstance(max_weight,int)) and max_weight>=self.api_table_ref_type['max_weight']['range'][0] and max_weight<=self.api_table_ref_type['max_weight']['range'][1]:self.max_weight = max_weight
        else: raise ValueError("max_weight must be float eg: 627 and must be in range [0,1000]")
        
        if isinstance(highest_drug_peak,self.api_table_ref_type['highest_drug_peak']['type']) or (min(highest_drug_peak)>self.api_table_ref_type['highest_drug_peak']['range'][0] and max(highest_drug_peak)<self.api_table_ref_type['highest_drug_peak']['range'][1]): self.highest_drug_peak = highest_drug_peak
        else:raise ValueError("highest drug peak must be list eg: [1003] and must be in range 200 and 2601")
        
        if isinstance(critical_peak,self.api_table_ref_type['critical_peak']['type']) or (min(critical_peak)>self.api_table_ref_type['critical_peak']['range'][0] and max(critical_peak)<self.api_table_ref_type['critical_peak']['range'][1]) : self.critical_peak = critical_peak      
        else: raise ValueError("critical peak must be list eg: [1002] and must be in range 200 and 2601")

        if isinstance(quantification_peak,self.api_table_ref_type['quantification_peak']['type']):
            if len(quantification_peak)!=0 and min(quantification_peak)>self.api_table_ref_type['quantification_peak']['range'][0] and max(quantification_peak)<self.api_table_ref_type['quantification_peak']['range'][1]: self.quantification_peak = quantification_peak
            else: self.quantification_peak = quantification_peak
        else:
            raise ValueError("quantification peak must be list and must be in range 200 and 2601")

        if isinstance(max_conc,self.api_table_ref_type['max_conc']['type']) and max_conc>0: self.max_conc = max_conc
        else: raise ValueError("max_conc must be numeric and must be more than 0")
        
        if isinstance(min_conc,self.api_table_ref_type['min_conc']['type']) and min_conc>0:self.min_conc = min_conc
        else: raise ValueError("min_conc must be numeric and must be more than 0")

        if isinstance(max_conc_ycal,self.api_table_ref_type['max_conc_ycal']['type']): self.max_conc_ycal=max_conc_ycal
        else: raise ValueError("max_conc_ycal must be in list")
        
        if isinstance(min_conc_ycal,self.api_table_ref_type['min_conc_ycal']['type']): self.min_conc_ycal=min_conc_ycal
        else: raise ValueError("min_conc_ycal must be in list")

        if isinstance(q_threshold,self.api_table_ref_type['q_threshold']['type']): self.q_threshold=q_threshold
        else: raise ValueError("q_threshold must be in list")

        if (isinstance(peak_threshold,self.api_table_ref_type['peak_threshold']['type']) or isinstance(peak_threshold,int)) and peak_threshold>=self.api_table_ref_type['peak_threshold']['range'][0] and peak_threshold<=self.api_table_ref_type['peak_threshold']['range'][1]:self.peak_threshold = peak_threshold
        else: raise ValueError("peak_threshold must be float eg: 0.04 and must be in range [0,1]")
        
        if (isinstance(pass_threshold,self.api_table_ref_type['pass_threshold']['type']) or isinstance(pass_threshold,int)) and pass_threshold>=self.api_table_ref_type['pass_threshold']['range'][0] and pass_threshold<=self.api_table_ref_type['pass_threshold']['range'][1] : self.pass_threshold = pass_threshold 
        else: raise ValueError("pass_threshold must be float eg: 0.7 and must be in range [0,1]")
        
        if (isinstance(drug_limit,self.api_table_ref_type['drug_limit']['type']) or isinstance(drug_limit,int)) and drug_limit>=self.api_table_ref_type['drug_limit']['range'][0] and drug_limit<=self.api_table_ref_type['drug_limit']['range'][1] : self.drug_limit = drug_limit
        else: raise ValueError("drug_limit must be float eg: 25 and must be in range [0,100]")


    def __repr__(self):
        return f"Api_info Table \ndrugname = {self.drugname}\npeaks = {self.peaks}\nmax_weight = {self.max_weight}\nhighest_drug_peak = {self.highest_drug_peak}\ncritical_peak = {self.critical_peak}\min_conc = {self.min_conc}\nmax_conc = {self.max_conc}\nmax_conc_ycal = {self.max_conc_ycal}nmin_conc_ycal = {self.min_conc_ycal}\nq_threshold = {self.q_threshold}\npeak_threshold = {self.peak_threshold}\npass_threshold = {self.pass_threshold}"
    
    def __str__(self):
        return f"Api_info Table \ndrugname = {self.drugname}\npeaks = {self.peaks}\nmax_weight = {self.max_weight}\nhighest_drug_peak = {self.highest_drug_peak}\ncritical_peak = {self.critical_peak}\min_conc = {self.min_conc}\nmax_conc = {self.max_conc}\nmax_conc_ycal = {self.max_conc_ycal}nmin_conc_ycal = {self.min_conc_ycal}\nq_threshold = {self.q_threshold}\npeak_threshold = {self.peak_threshold}\npass_threshold = {self.pass_threshold}"
    
    
    def view_instance(self):
        print(f"Api_info Table \ndrugname = {self.drugname}\npeaks = {self.peaks}\nmax_weight = {self.max_weight}\nhighest_drug_peak = {self.highest_drug_peak}\ncritical_peak = {self.critical_peak}\min_conc = {self.min_conc}\nmax_conc = {self.max_conc}\nmax_conc_ycal = {self.max_conc_ycal}\nmin_conc_ycal = {self.min_conc_ycal}\nq_threshold = {self.q_threshold}\npeak_threshold = {self.peak_threshold}\npass_threshold = {self.pass_threshold}")
    
    def verify_insertion_type(self,cun):
        row = cun.execute(f"SELECT * FROM {self.table_name} WHERE drugname = '{self.drugname}'").fetchone()#[0]
        keys = self.api_table_ref_type.keys()
        for num,r in enumerate(zip(row,keys)):
            if r[1] in ['highest_drug_peak',"peaks","critical_peak"]:
                if ((isinstance(eval(r[0]),self.api_table_ref_type[r[1]]['type'])) or ((min(eval(r[0]))>=self.api_table_ref_type[r[1]]['range'][0] and max(eval(r[0]))<=self.api_table_ref_type[r[1]]['range'][1])))==False:
                    raise ValueError(f"Not matching the requirement for {r[1]} expecting = {self.api_table_ref_type[r[1]]['type']}")
            
            elif r[1] in ['peak_threshold','pass_threshold','drug_limit']:
                if (isinstance(r[0],self.api_table_ref_type[r[1]]['type']) or isinstance(r[0],int)) and (r[0]>=self.api_table_ref_type[r[1]]['range'][0] and r[0]<=self.api_table_ref_type[r[1]]['range'][1])==False:
                    raise ValueError(f"Not matching the requirement for {r[1]} expecting = {self.api_table_ref_type[r[1]]['type']}")
            else:
                if (isinstance(r[0],self.api_table_ref_type[r[1]]['type']))==False:
                    raise ValueError(f"Not matching the requirement for {r[1]} expecting = {self.api_table_ref_type[r[1]]['type']}")
                    
            
    
    def insert_to_db(self,view=True):
        """
        Inserted the data to table for current instance, verify the instance by using view_instance
        view: used to view the data of instance before inserting to table
        """
        cun = self.database.cursor()
        if view: self.view_instance()
        try:
            print(f"Trying to insert {self.drugname} to {self.table_name} table")
            cun.execute(f"INSERT INTO {self.table_name} (drugname,peaks,max_weight,highest_drug_peak,critical_peak,quantification_peak,min_conc,max_conc,min_conc_ycal,max_conc_ycal,q_threshold,peak_threshold,pass_threshold,drug_limit) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(self.drugname,str(list(self.peaks)),self.max_weight,str(list(self.highest_drug_peak)),str(list(self.critical_peak)),str(list(self.quantification_peak)),self.min_conc,self.max_conc,str(list(self.min_conc_ycal)),str(list(self.max_conc_ycal)),self.q_threshold,self.peak_threshold,self.pass_threshold,self.drug_limit));
            print("\nVerification of type is in process.....\n")
            self.verify_insertion_type(cun)
            print("Verfied Successfully")
            inp = input("\nEnter yes or y to commit on database...\n ")
            if str(inp).lower()=='yes' or str(inp).lower() == 'y':
                self.database.commit()
                print(f"{self.drugname} is inserted to {self.table_name} table")
            else:
                print("\nAborted the commit\n")
        except Exception as e:
            self.database.rollback()
            print(f"Failed to insert {self.drugname} to {self.table_name} by {e}")
        finally:
            super().close_database()
     
    @staticmethod
    def update_row(dbpath,drugname,column,value):
        """
        Update the value for specific column in api_info columns
        """
        database = Database_OAK(dbpath)
        cur = database.database.cursor()
        
        try:
            print("Verifying process started...")
            print("Verified beforing updating...")
            cur.execute(f'''UPDATE api_info SET {column} = '{value}' WHERE drugname='{drugname}';''')
            #verifying_Query(cur,drugname,database)
            print("Verified after updated")
            inp = input("\nEnter yes or y to commit on database...\n ")
            if str(inp).lower()=='yes' or str(inp).lower() == 'y':
                                database.database.commit()
                                print(f"{column} updated for {drugname} by {value}")
            else:
                                print("\nAborted the commit\n")
                    
            
        except Exception as e:
            database.database.rollback()
            print(f"Failed to update {column} for {drugname} by {e}")
        finally:
            database.close_database()
            del cur,database
            
    @staticmethod
    def delete_row(dbpath, drugname):
        """
        Delete a row with the given drugname from the api_info table.
        """
        database = Database_OAK(dbpath)
        cur = database.database.cursor()

        try:
            print("Deleting row from api_info...")
            cur.execute(f"DELETE FROM api_info WHERE drugname = '{drugname}';")
            # verifying_Query(cur,drugname,database)
            print("Verified after deleting")
            inp = input("\nEnter yes or y to commit on database...\n ")
            if str(inp).lower() == 'yes' or str(inp).lower() == 'y':
                database.database.commit()
                print(f"{drugname} row deleted from api_info table.")
            else:
                print("\nAborted the commit\n")

        except Exception as e:
            database.database.rollback()
            print(f"Failed to delete {drugname} row from api_info table by {e}")
        finally:
            database.close_database()
            del cur, database           
        
    @staticmethod
    def fetch_data(dbpath):
        """
        fetch the entire data from table api_info of Oak_database
        """
        import pandas as pd
        database = Database_OAK(dbpath)
        df =  pd.read_sql("select * from api_info",database.database)
        for col in df.columns:
            if database.api_table_ref_type[col]['type']==list:
                df[col]= df[col].apply(lambda x: eval(x))  
        database.close_database()
        return df
    
#######################################################################################################################################

class Pharmacy_info(Database_OAK):
    """
    Used to perform operation of pharmacy_info of OAK Database
    Attributes:
        drugname(str): "drugname of new sample"
        key(str) : "type drug or solvent"
        xcal_pure : "pure xcal data"
        peaks(list): "list of drug peaks eg [1233,1344,1634]"
        peak_threshold(float): default(0.04), peak identification limit for drug that is promience for peak finding in quantification process
        quantification_peak(list): default [], quantification peak used to quantify in drug
        critical_peak(list): default [], critical peak to be getting in drug
        highest_conc(int): highest concentration drug limit
        lowest_conc(int): lowest concentartion drug limit
        solvents(list): list of solvents that specific drug as been seen
        pass_threshold(float): default(0.7), pass_threshold for drug that can be consider as pass_threshold as pass that is 0.7 for structure similiarity q_identification
        verified("string"): default (No), if yes the drug is authenticated for identification process
        pharmacy_table_ref_type: reference for respective attribute and there type and there range
    
    STATEMENT:
        CREATE TABLE "pharmacy_info" (
        "drugname"	TEXT NOT NULL,
        "key"	TEXT,
        "xcal_pure" BLOB,
        "peaks"	BLOB,
        "peak_threshold"	NUMERIC,
        "quantification_peak"	BLOB,
        "critical_peak"	BLOB,
        "highest_conc"	NUMERIC,
        "lowest_conc"	NUMERIC,
        "solvents"	BLOB,
        "pass_threshold"	NUMERIC DEFAULT 0.7,
        "verified"	TEXT,
        PRIMARY KEY("drugname")
    )
    """
    
    def __init__(self,dbpath,
                 drugname:str,
                 key:str,
                 xcal_pure :list=[],
                 peaks: list=[],
                 peak_threshold: float=0.04,
                 quantification_peak: list=[],
                 critical_peak: list=[],
                 highest_conc: int = 20,
                 lowest_conc: int =  1,
                 solvents: list = ['water'],
                 pass_threshold: float = 0.7,
                 verified: str = 'No'
                ):
        """
            dbpath(*.db): "path of sql database path"
            drugname(str): "drugname of new sample"
            key(str) : "type drug or solvent"
            xcal_pure(list) : "pure xcal data"
            peaks(list): "list of drug peaks eg [1233,1344,1634]"
            peak_threshold(float): default(0.04), peak identification limit for drug that is promience for peak finding in q_identification process
            quantification_peak(list): default [], quantification peak used to quantify in drug
            critical_peak(list): default [], critical peak to be getting in drug
            highest_conc(int): default 20, highest drug peak of drug sample
            lowest_conc(int): default 1, lowest drug peak of drug sample
            solvents(list): default [], solvents in sample
            pass_threshold(float): default(0.7), pass_threshold for drug that can be consider as pass_threshold as pass that is 0.7
            verified("string"): default (No), if yes the drug is authenticated for identification process
        """
        
        super().__init__(dbpath)  
        self.table_name = "pharmacy_info"
        self.dbpath = dbpath
        self.drugname = str(drugname).capitalize()
        self.key = str(key).capitalize()
        
        if isinstance(xcal_pure,self.pharmacy_table_ref_type['xcal_pure']['type']): self.xcal_pure=xcal_pure
        else: raise ValueError("xcal_pure must be in list")
             
        if isinstance(peaks,self.pharmacy_table_ref_type['peaks']['type']) and min(peaks)>self.pharmacy_table_ref_type['peaks']['range'][0] and max(peaks)<self.pharmacy_table_ref_type['peaks']['range'][1]: self.peaks = peaks
        else: raise ValueError("peaks must be list eg: [1002,1408,1038] and must be in range 200 and 2601")
            
        if isinstance(quantification_peak,self.pharmacy_table_ref_type['quantification_peak']['type']):
            if len(quantification_peak)!=0 and min(quantification_peak)>self.pharmacy_table_ref_type['quantification_peak']['range'][0] and max(quantification_peak)<self.pharmacy_table_ref_type['quantification_peak']['range'][1]: self.quantification_peak = quantification_peak
            else: self.quantification_peak = quantification_peak
        else:
            raise ValueError("quantification peak must be list and must be in range 200 and 2601")
            
        if isinstance(critical_peak,self.pharmacy_table_ref_type['critical_peak']['type']):
            if len(critical_peak)!=0 and min(critical_peak)>self.pharmacy_table_ref_type['critical_peak']['range'][0] and max(critical_peak)<self.pharmacy_table_ref_type['critical_peak']['range'][1]: self.critical_peak = critical_peak
            else: self.critical_peak = critical_peak
        else:
            raise ValueError("critical peak must be list and must be in range 200 and 2601")
            
        if isinstance(highest_conc,self.pharmacy_table_ref_type['highest_conc']['type']) and highest_conc>0: self.highest_conc = highest_conc
        else: raise ValueError("highest_conc must be int and must be more than 0")
        
        if isinstance(lowest_conc,self.pharmacy_table_ref_type['lowest_conc']['type']) and lowest_conc>0:self.lowest_conc = lowest_conc
        else: raise ValueError("lowest_conc must be int and must be more than 0")
        
        if isinstance(solvents,self.pharmacy_table_ref_type['solvents']['type']): self.solvents=solvents
        else: raise ValueError("solvents must be in list")
            
        if (isinstance(peak_threshold,self.pharmacy_table_ref_type['peak_threshold']['type']) or isinstance(peak_threshold,int)) and peak_threshold>=self.pharmacy_table_ref_type['peak_threshold']['range'][0] and peak_threshold<=self.pharmacy_table_ref_type['peak_threshold']['range'][1]:self.peak_threshold = peak_threshold
        else: raise ValueError("peak_threshold must be float eg: 0.04 and must be in range [0,1]")
        
        if (isinstance(pass_threshold,self.pharmacy_table_ref_type['pass_threshold']['type']) or isinstance(pass_threshold,int)) and pass_threshold>=self.pharmacy_table_ref_type['pass_threshold']['range'][0] and pass_threshold<=self.pharmacy_table_ref_type['pass_threshold']['range'][1] : self.pass_threshold = pass_threshold 
        else: raise ValueError("pass_threshold must be float eg: 0.7 and must be in range [0,1]")
    
        self.verified = str(verified).capitalize() if str(verified).lower() in ['yes','no'] else "No"
        
    def __repr__(self):
        return f"Pharmacy_Info Table \ndrugname = {self.drugname}\nkey = {self.key}\nxcal_pure = {self.xcal_pure}\npeaks = {self.peaks}\nhighest_conc = {self.highest_conc}\nlowest_conc = {self.lowest_conc}\nsolvents = {self.solvents}\npeak_threshold = {self.peak_threshold}\npass_threshold = {self.pass_threshold}\nverified = {self.verified}"
    
    def __str__(self):
        return f"Pharmacy_Info Table \ndrugname = {self.drugname}\nkey = {self.key}\nxcal_pure = {self.xcal_pure}\npeaks = {self.peaks}\nhighest_conc = {self.highest_conc}\nlowest_conc = {self.lowest_conc}\nsolvents = {self.solvents}\npeak_threshold = {self.peak_threshold}\npass_threshold = {self.pass_threshold}\nverified = {self.verified}"
    
    
    def view_instance(self):
        print(f"Pharmacy_Info Table \ndrugname = {self.drugname}\nkey = {self.key}\nxcal_pure = {self.xcal_pure}\npeaks = {self.peaks}\nhighest_conc = {self.highest_conc}\nlowest_conc = {self.lowest_conc}\nsolvents = {self.solvents}\npeak_threshold = {self.peak_threshold}\npass_threshold = {self.pass_threshold}\nverified = {self.verified}")
    
    
    
    def insert_to_db(self,view=True):
        """
        Inserted the data to table for current instance, verify the instance by using view_instance
        view: used to view the data of instance before inserting to table
        """
        cun = self.database.cursor()
        if view: self.view_instance()
        try:
            print(f"Trying to insert {self.drugname} to {self.table_name} table")
            cun.execute(f"INSERT INTO {self.table_name} (drugname,key,xcal_pure,peaks,peak_threshold,quantification_peak,critical_peak,highest_conc,lowest_conc,solvents,pass_threshold,verified) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",(self.drugname,self.key,str(list(self.xcal_pure)),str(list(self.peaks)),self.peak_threshold,str(list(self.quantification_peak)),str(list(self.critical_peak)),self.highest_conc,self.lowest_conc,str(self.solvents),self.pass_threshold,self.verified));
            print("\nVerification of type is in process.....\n")
            #self.verify_insertion_type(cun)
            print("Verfied Successfully")
            inp = input("\nEnter yes or y to commit on database...\n ")
            if str(inp).lower()=='yes' or str(inp).lower() == 'y':
                self.database.commit()
                print(f"{self.drugname} is inserted to {self.table_name} table")
            else:
                print("\nAborted the commit\n")
        except Exception as e:
            self.database.rollback()
            print(f"Failed to insert {self.drugname} to {self.table_name} by {e}")
        finally:
            super().close_database()

    @staticmethod
    def update_row(dbpath,drugname,column,value):
        """
        Update the value for specific column in pharmacy_info columns
        """
        database = Database_OAK(dbpath)
        cur = database.database.cursor()
        
        try:
            print("Verifying process started...")
            print("Verified beforing updating...")
            cur.execute(f'''UPDATE pharmacy_info SET {column} = '{value}' WHERE drugname='{drugname}';''')
            #verifying_Query(cur,drugname,database)
            print("Verified after updated")
            inp = input("\nEnter yes or y to commit on database...\n ")
            if str(inp).lower()=='yes' or str(inp).lower() == 'y':
                                database.database.commit()
                                print(f"{column} updated for {drugname} by {value}")
            else:
                                print("\nAborted the commit\n")
                    
            
        except Exception as e:
            database.database.rollback()
            print(f"Failed to update {column} for {drugname} by {e}")
        finally:
            database.close_database()
            del cur,database

    @staticmethod
    def delete_row(dbpath, drugname):
        """
        Delete a row with the given drugname from the pharmacy_info table.
        """
        database = Database_OAK(dbpath)
        cur = database.database.cursor()

        try:
            print("Deleting row from pharmacy_info...")
            cur.execute(f"DELETE FROM pharmacy_info WHERE drugname = '{drugname}';")
            # verifying_Query(cur,drugname,database)
            print("Verified after deleting")
            inp = input("\nEnter yes or y to commit on database...\n ")
            if str(inp).lower() == 'yes' or str(inp).lower() == 'y':
                database.database.commit()
                print(f"{drugname} row deleted from pharmacy_info table.")
            else:
                print("\nAborted the commit\n")

        except Exception as e:
            database.database.rollback()
            print(f"Failed to delete {drugname} row from pharmacy_info table by {e}")
        finally:
            database.close_database()
            del cur, database    
            
    @staticmethod
    def fetch_data(dbpath):
        """
        fetch the entire data from table pharmacy_info of Oak_database
        """
        import pandas as pd
        database = Database_OAK(dbpath)
        dk =  pd.read_sql("select * from pharmacy_info",database.database)
        for col in dk.columns:
            if database.pharmacy_table_ref_type[col]['type']==list:
                dk[col]=dk[col].apply(lambda x: eval(x))
        database.close_database()
        return dk


###############################################################################################################
class Syringe_info(Database_OAK):
    """
    Used to perform insert data to Syringe_info table of Oak_Database
    Attributes:
        dbpath(*.db): "path of sql database path"
        name(str): "name of new sample"
        peaks(list): "list of peaks"
        file(str): "path of json record with xcal,ycal,raw" or file(dict): "with xcal,yal,raw as keys and values of spectrum of xcal,ycal,raw respectively"
        syringe_info_ref_type: reference for respective attribute and there type and size
        kwargs looks for xcal ycal raw 
    methods:
        view_instance used to plot data of file provide at creating an instance
        insert_to_db used to insert data to table 
        fetch_data returns the rows present in table
        
        CREATE TABLE "syringe_info" (
        "name"	TEXT NOT NULL,
        "peaks"	BLOB,
        "xcal"	BLOB,
        "ycal"	BLOB,
        "raw"	BLOB,
        PRIMARY KEY("name")
    )
    """
    def __init__(self,dbpath,name:str,peaks:list=[],file=None,apply_peak_normalization=True,**kwargs):
        """
        dbpath(*.db): "path of sql database path"
        name(str): "name of new sample"
        file(str): "path of json record with xcal,ycal,raw" or file(dict): "with xcal,yal,raw as keys and values of spectrum of xcal,ycal,raw respectively"
        kwargs looks for xcal ycal raw     
        """
        super().__init__(dbpath)
        self.table_name = 'syringe_info'
        self.dbpath = dbpath
        self.name = str(name).capitalize()
        
        if isinstance(file,str) and file.split('.')[-1]=='json':
            data = json.load(open(file))
            if apply_peak_normalization:
                self.xcal = peak_normalization(data['xcal'])
                self.ycal = peak_normalization(data['ycal'])
                self.raw = peak_normalization(data['raw'])
            else:
                self.xcal = data['xcal']
                self.ycal = data['ycal']
                self.raw = data['raw']     
        elif isinstance(file,dict):
            assert(sorted(file.keys()) == ['raw','xcal','ycal'])
            if apply_peak_normalization:
                self.xcal = peak_normalization(data['xcal'])
                self.ycal = peak_normalization(data['ycal'])
                self.raw = peak_normalization(data['raw'])
            else:
                self.xcal = data['xcal']
                self.ycal = data['ycal']
                self.raw = data['raw'] 
        elif file is None:
            if apply_peak_normalization:
                self.xcal = peak_normalization(kwargs['xcal'])
                self.ycal = peak_normalization(kwargs['ycal'])
                self.raw = peak_normalization(kwargs['raw'])
            else:
                self.xcal = kwargs['xcal']
                self.ycal = kwargs['ycal']
                self.raw = kwargs['raw']

        if isinstance(peaks,self.syringe_info_ref_type['peaks']['type']):
            if len(peaks)>0 and min(peaks)>self.syringe_info_ref_type['peaks']['range'][0] and max(peaks)<self.syringe_info_ref_type['peaks']['range'][1]:
                self.peaks = peaks
            else:
                self.peaks = peaks
            
    def __repr__(self):
        return f"Structure Table\ninstance = {self.name},\nlength of xcal = {len(self.xcal)},\nlength of ycal = {len(self.ycal)},\nlength of raw = {len(self.raw)}\nPeaks = {self.peaks}"
    
    def __str__(self):
        return f"Structure Table\ninstance = {self.name},\nlength of xcal = {len(self.xcal)},\nlength of ycal = {len(self.ycal)},\nlength of raw = {len(self.raw)}\nPeaks = {self.peaks}"       
    
 
    def view_instance(self):
        """
        this method is used to plot xcal,ycal,raw of the instance
        """
        fig,ax = plt.subplots(ncols=3,nrows=1,figsize=(12,4))
        for num,level in enumerate(zip(['xcal','ycal','raw'],[self.xcal,self.ycal,self.raw])):

            ax[num].plot(level[1])
            ax[num].set_title(level[0])
        plt.show()
              
    def verify_insertion_type(self,cun):
        row = cun.execute(f"SELECT * FROM {self.table_name} WHERE name = '{self.name}'").fetchone()#[0]
        keys =  self.syringe_info_ref_type.keys()
        for r in zip(row,keys):
            if r[1]!='drugname':
                if (isinstance(eval(r[0]),self.syringe_info_ref_type[r[1]]['type']) and len(eval(r[0]))==self.syringe_info_ref_type[r[1]]['size']) ==False:
                    raise ValueError("Not matching the requirement")    
            else:
                if isinstance(r[0],self.syringe_info_ref_type[r[1]]['type'])==False:
                    raise ValueError("Not matching the requirement") 
                
            
    def insert_to_db(self,view=True):
        """
        Inserted the data to table for current instance, verify the instance by using view_instance
        view: used to view the data of instance before inserting to table
        """
        cun = self.database.cursor()
        if view:
            self.view_instance()
        try:
            print(f"Trying to insert {self.name} to {self.table_name} table")
            cun.execute(f"INSERT INTO {self.table_name} (name,peaks,xcal,ycal,raw) VALUES(?,?,?,?,?)",(self.name,str(list(self.peaks)),str(list(self.xcal)),str(list(self.ycal)),str(list(self.raw))));
            print("\nVerification of type is in process.....\n")
            #self.verify_insertion_type(cun)
            print("Verfied Successfully")
            inp = input("\nEnter yes or y to commit on database...\n ")
            if str(inp).lower()=='yes' or str(inp).lower() == 'y':
                self.database.commit()
                print(f"{self.name} is inserted to {self.table_name} table")
            else:
                print("\nAborted the commit\n")
        except Exception as e:
            self.database.rollback()
            print("Failed to insert {self.drugname} to {self.table_name} by {e}")
        finally:
            super().close_database()
            
    @staticmethod
    def fetch_data(dbpath):
        """
        fetch the entire data from table syringe_info of Oak_database
        """
        import pandas as pd
        database = Database_OAK(dbpath)
        df =  pd.read_sql("select * from syringe_info",database.database)
        for col in df.columns:
            if database.syringe_info_ref_type[col]['type']==list:
                df[col]= df[col].apply(lambda x: eval(x))     
        database.close_database()
        return df
    
    @staticmethod
    def update_row(dbpath,name,column,value):
        """
        Update the value for specific column in syringe_info columns
        """
        database = Database_OAK(dbpath)
        cur = database.database.cursor()
        
        try:
            print("Verifying process started...")
            print("Verified beforing updating...")
            cur.execute(f'''UPDATE syringe_info SET {column} = '{value}' WHERE name='{name}';''')
            #verifying_Query(cur,name,database)
            print("Verified after updated")
            inp = input("\nEnter yes or y to commit on database...\n ")
            if str(inp).lower()=='yes' or str(inp).lower() == 'y':
                                database.database.commit()
                                print(f"{column} updated for {name} by {value}")
            else:
                                print("\nAborted the commit\n")
                    
            
        except Exception as e:
            database.database.rollback()
            print(f"Failed to update {column} for {name} by {e}")
        finally:
            database.close_database()
            del cur,database

    @staticmethod
    def delete_row(dbpath, name):
        """
        Delete a row with the given name from the syringe_info table.
        """
        database = Database_OAK(dbpath)
        cur = database.database.cursor()

        try:
            print("Deleting row from syringe_info...")
            cur.execute(f"DELETE FROM syringe_info WHERE name = '{name}';")
            # verifying_Query(cur,name,database)
            print("Verified after deleting")
            inp = input("\nEnter yes or y to commit on database...\n ")
            if str(inp).lower() == 'yes' or str(inp).lower() == 'y':
                database.database.commit()
                print(f"{name} row deleted from syringe_info table.")
            else:
                print("\nAborted the commit\n")

        except Exception as e:
            database.database.rollback()
            print(f"Failed to delete {name} row from syringe_info table by {e}")
        finally:
            database.close_database()
            del cur, database
###############################################################################################################    
class Solvents_info(Database_OAK):
    """
    Used to retrive cutting subtance, solvents info
    Attributes:
        drugname(str): "drugname of new sample"
        peaks(list): "list of drug peaks eg [1233,1344,1634]"
        window_size(int): "window size of the finding the peaks within + or - of window size"
        solvents_table_ref_type: reference for respective attribute and there type and there range
    
    
    
    STATEMENT:
        CREATE TABLE "solvents_info" (
        "name"	TEXT NOT NULL,
        "peaks"	BLOB,
        "window_size"	INTEGER NOT NULL DEFAULT 10,
        PRIMARY KEY("name")
    )
    """
    
    def __init__(self,dbpath,name:str,peaks:list=[],window_size:int=10):
        """
        dbpath(*.db): "path of sql database path"
        name(str): "drugname of new sample"
        peaks(list): "list of drug peaks eg [1233,1344,1634]"
        window_size(int)(default: 10): windows size to find solvent peaks in sample 
        """
        super().__init__(dbpath)
        self.table_name = "solvents_info"
        self.dbpath = dbpath
        self.name = str(name).capitalize()
        
            
        if isinstance(peaks,self.solvents_table_ref_type['peaks']['type']):
            if len(peaks)>0 and min(peaks)>self.solvents_table_ref_type['peaks']['range'][0] and max(peaks)<self.solvents_table_ref_type['peaks']['range'][1]:
                self.peaks = peaks
            else:
                self.peaks = peaks
        
        if isinstance(window_size,self.solvents_table_ref_type['window_size']['type']): self.window_size = window_size
        else: self.window_size = window_size   
    
    def __repr__(self):
        return f"{self.table_name} Table \nname = {self.name}\npeaks = {self.peaks}\nwindow_size = {self.window_size}\n"
    
    def __str__(self):
        return f"{self.table_name} Table \nname = {self.name}\npeaks = {self.peaks}\nwindow_size  = {self.window_size}\n"
    
    def view_instance(self):
        print(f"{self.table_name} Table \nname = {self.name}\npeaks = {self.peaks}\nwindow_size  = {self.window_size}\n")
        
    
    def insert_to_db(self,view=True):
        """
        Inserted the data to table for current instance, verify the instance by using view_instance
        view: used to view the data of instance before inserting to table
        """
        cun = self.database.cursor()
        if view: self.view_instance()
        try:
            print(f"Trying to insert {self.name} to {self.table_name} table")
            cun.execute(f"INSERT INTO {self.table_name} (name,peaks,window_size) VALUES(?,?,?)",(self.name,str(list(self.peaks)),self.window_size));
            print("\nVerification of type is in process.....\n")
            #self.verify_insertion_type(cun)
            print("Verfied Successfully")
            inp = input("\nEnter yes or y to commit on database...\n ")
            if str(inp).lower()=='yes' or str(inp).lower() == 'y':
                self.database.commit()
                print(f"{self.name} is inserted to {self.table_name} table")
            else:
                print("\nAborted the commit\n")
        except Exception as e:
            self.database.rollback()
            print(f"Failed to insert {self.name} to {self.table_name} by {e}")
        finally:
            super().close_database()
            
    @staticmethod
    def update_row(dbpath,drugname,column,value):
        """
        Update the value for specific column in solvents_info columns
        """
        database = Database_OAK(dbpath)
        cur = database.database.cursor()
        
        try:
            print("Verifying process started...")
            print("Verified beforing updating...")
            cur.execute(f'''UPDATE solvents_info SET {column} = '{value}' WHERE name='{drugname}';''')
            #verifying_Query(cur,drugname,database)
            print("Verified after updated")
            inp = input("\nEnter yes or y to commit on database...\n ")
            if str(inp).lower()=='yes' or str(inp).lower() == 'y':
                                database.database.commit()
                                print(f"{column} updated for {drugname} by {value}")
            else:
                                print("\nAborted the commit\n")
                    
            
        except Exception as e:
            database.database.rollback()
            print(f"Failed to update {column} for {drugname} by {e}")
        finally:
            database.close_database()
            del cur,database

    @staticmethod
    def delete_row(dbpath, name):
        """
        Delete a row with the given solvent from the solvents_info table.
        """
        database = Database_OAK(dbpath)
        cur = database.database.cursor()

        try:
            print("Deleting row from solvents_info...")
            cur.execute(f"DELETE FROM solvents_info WHERE name = '{name}';")
            # verifying_Query(cur,name,database)
            print("Verified after deleting")
            inp = input("\nEnter yes or y to commit on database...\n ")
            if str(inp).lower() == 'yes' or str(inp).lower() == 'y':
                database.database.commit()
                print(f"{name} row deleted from solvents_info table.")
            else:
                print("\nAborted the commit\n")

        except Exception as e:
            database.database.rollback()
            print(f"Failed to delete {name} row from solvents_info table by {e}")
        finally:
            database.close_database()
            del cur, database
            
    @staticmethod
    def fetch_data(dbpath):
        """
        fetch the entire data from table solvents_info of Oak_database
        """
        database = Database_OAK(dbpath)
        df =  pd.read_sql("select * from solvents_info",database.database)
        for col in df.columns:
            if database.solvents_table_ref_type[col]['type']==list:
                df[col]=df[col].apply(lambda x: eval(x))
        database.close_database()
        return df

############################################################################################################

class ID2Database:
    def __init__(self, db_path):
        """
        Initialize the database connection class.
        :param db_path: Path to the database directory.
        """
        self.db_path = db_path
        self.db_file = os.path.join(self.db_path, "ID2.db")

    def connect(self):
        """
        Establish a connection to the SQLite database.
        :return: SQLite connection object.
        """
        if not os.path.exists(self.db_file):
            raise FileNotFoundError(f"Database file not found at {self.db_file}")
        # print(f"Connecting to SQLite database at {self.db_file}")
        return sqlite3.connect(self.db_file)

    def fetch_data(self, query, params=None):
        """
        Execute a query and fetch data from the database.
        :return: List of rows as dictionaries.
        """
        connection = self.connect()
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()
        
        try:
            cursor.execute(query, params or ())
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            print(f"Error fetching data: {e}")
            raise
        finally:
            connection.close()

    def fetch_all_data(self, table_name='ID2_Reference'):
        """
        Fetch all data from a specified table.
        :param table_name: Name of the table.
        :return: List of rows as dictionaries.
        """
        query = f"SELECT * FROM {table_name};"
        return self.fetch_data(query)
    

    def get_row_count(self, table_name='ID2_Reference'):
        """
        Get the total number of rows in a specified table.
        :param table_name: Name of the table.
        :return: Row count as an integer.
        """
        query = f"SELECT COUNT(*) as count FROM {table_name};"
        result = self.fetch_data(query)
        return result[0]["count"] if result else 0

    def fetch_data_by_id(self, table_name='ID2_Reference', record_id=None):
        """
        Fetch data from a specified table by a given ID.
        :param table_name: Name of the table.
        :param record_id: The specific ID to filter by.
        :return: List of rows as dictionaries.
        """
        if record_id is None:
            raise ValueError("Record ID must be provided to fetch data.")
        
        query = f"SELECT * FROM {table_name} WHERE ID = ?;"
        return self.fetch_data(query, params=(record_id,))