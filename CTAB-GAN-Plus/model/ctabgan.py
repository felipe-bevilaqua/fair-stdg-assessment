"""
Generative model training algorithm based on the CTABGANSynthesiser

"""
import pandas as pd
import time
from model.pipeline.data_preparation import DataPrep
from model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer

import warnings

warnings.filterwarnings("ignore")

class CTABGAN():

    def __init__(self,
                 #raw_csv_path = "Real_Datasets/Adult.csv",
                 #test_ratio = 0.20,
                 categorical_columns = [ 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income'], 
                 log_columns = [],
                 mixed_columns= {'capital-loss':[0.0],'capital-gain':[0.0]},
                 general_columns = ["age"],
                 non_categorical_columns = [],
                 integer_columns = ['age', 'fnlwgt','capital-gain', 'capital-loss','hours-per-week'],
                 problem_type= {"Classification": "income"},
                 lr = 2e-4,
                 epochs = 150,
                 batch_size = 500):

        self.__name__ = 'CTABGAN'
              
        
        #self.raw_df = pd.read_csv(raw_csv_path)
        #self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.general_columns = general_columns
        self.non_categorical_columns = non_categorical_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.synthesizer = CTABGANSynthesizer()
                
    def fit(self,
            df,
            test_ratio =0.0,
            categorical_columns = [ 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income'], 
            log_columns = [],
            mixed_columns= {'capital-loss':[0.0],'capital-gain':[0.0]},
            general_columns = ["age"],
            non_categorical_columns = [],
            integer_columns = ['age', 'fnlwgt','capital-gain', 'capital-loss','hours-per-week'],
            problem_type= {"Classification": "income"},):
        
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.general_columns = general_columns
        self.non_categorical_columns = non_categorical_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type
        self.test_ratio = test_ratio
        
        start_time = time.time()
        self.data_prep = DataPrep(df ,self.categorical_columns,self.log_columns,self.mixed_columns,self.general_columns,self.non_categorical_columns,self.integer_columns,self.problem_type,self.test_ratio)
        print(self.lr, self.batch_size)
        self.synthesizer.fit(train_data=self.data_prep.df, categorical = self.data_prep.column_types["categorical"], mixed = self.data_prep.column_types["mixed"],
        general = self.data_prep.column_types["general"], non_categorical = self.data_prep.column_types["non_categorical"], type=self.problem_type,
        epochs=self.epochs, batch_size=self.batch_size, lr=self.lr)
        
        end_time = time.time()
        print('Finished training in',end_time-start_time," seconds.")


    def generate_samples(self, sample_size):
        
        sample = self.synthesizer.sample(sample_size) 
        sample_df = self.data_prep.inverse_prep(sample)
        
        return sample_df
