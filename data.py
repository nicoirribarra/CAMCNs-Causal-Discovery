import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
drive.mount('/content/gdrive')

class DataProcess():
    def __init__(self, file_path_data, file_path_eval):

        self.df = pd.read_excel(file_path_data)
        self.eval_data = pd.read_csv(file_path_eval, header=None)
    
    def df_to_dict(self, df):
        aux = {}
        for i in range(len(df)):
            a = (df.iloc[i,1], df.iloc[i,0])
            aux[a] = df.iloc[i,2]
        return aux
    
    def prepare_data(self):
        # normalized the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.df.pop("Unnamed: 0")
        scaled = scaler.fit_transform(self.df)

        # create datefrane fir scaled data
        scaled_df=pd.DataFrame(data=scaled, columns=self.df.columns)
        data = self.df.copy()
        data[scaled_df.columns]=np.array(scaled_df)
        
        self.eval_data.pop(0)
        self.eval_data = self.eval_data.drop(self.eval_data.index[[0]])

        eval_dic = self.df_to_dict(self.eval_data)

        return data, eval_dic