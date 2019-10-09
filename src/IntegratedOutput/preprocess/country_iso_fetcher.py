from collections import defaultdict
import pandas as pd

class country_iso_code_fetcher:
    def __init__(self):
        self.SourceFile = '../../../WWF_Domain_Data_v1/CountryData/country_IsoCode.csv'
        self.df = pd.read_csv(self.SourceFile,index_col=0)
        self.iso_code_dict = defaultdict(lambda: None)
        for i,row in self.df.iterrows():
            self.iso_code_dict [row['country_name']] = row['iso_code']

        return

    def get_iso_code(self,c_name):
        return self.iso_code_dict[c_name]

ISO_CODE_OBJ = country_iso_code_fetcher()