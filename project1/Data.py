import numpy as np
import pandas as pd
import xlrd
from pathlib import Path

from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class Data:
    # region class members
    __filename = Path(__file__).parent / './ressources/forestfires.csv'

    __months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    __week_days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    # endregion

    def __init__(self):
        # region fetch data from .csv file
        df = pd.read_csv(self.__filename)
        self.__raw_data = df.values
        # endregion

        cols = range(0, 13)
        self.x = np.asarray(self.__raw_data[:, cols])

        self.x_no_label = self.x

        # region create labeling
        labels = np.asarray(self.x[:, 12])
        self.df = np.asarray(self.__raw_data[:, cols])
        np.random.shuffle(self.df)
        self.x = self.df[:,:12]

        
        labels = np.asarray(self.df[:,12])

        self.y1 = np.asarray(labels.astype('float64'))
        
        temp = []
        for label in labels:
            if label == 0:
                temp.append(0)
            else:
                temp.append(1)

        labels = temp
        self.names = sorted(set(labels))
        self.class_dict = dict(zip(self.names, (range(len(self.names)))))

        self.df[:, 12] = labels
        self.x3 = self.df
        self.y = np.asarray(([self.class_dict[value] for value in labels]))
        # endregion

        # region encoding months and week days
        self.month_dict = dict(zip(self.__months, range(1, len(self.__months) + 1)))
        self.week_days_dict = dict(zip(self.__week_days, range(1, len(self.__week_days) + 1)))
        
        self.month_enc = dict(zip(self.__months, 
                                  ['000000000001',
                                    '000000000010',
                                    '000000000100',
                                    '000000001000',
                                    '000000010000',
                                    '000000100000',
                                    '000001000000',
                                    '000010000000',
                                    '000100000000',
                                    '001000000000',
                                    '010000000000',
                                    '100000000000']))
        self.week_days_enc = dict(zip(self.__week_days,
                                      ['0000001',
                                        '0000010',
                                        '0000100',
                                        '0001000',
                                        '0010000',
                                        '0100000',
                                        '1000000']))

        aplic_cols = range(2, 4)
        vals = np.asarray(self.__raw_data[:, aplic_cols])

        for val in vals:
            val[0] = self.month_dict[val[0]]
            val[1] = self.week_days_dict[val[1]]
        self.x[:, aplic_cols] = vals
        
        # for val in vals:
        #     val[0] = self.month_enc[val[0]]
        #     val[1] = self.week_days_enc[val[1]]
        # self.x[:, aplic_cols] = vals
        # # endregion
        
        self.x = np.asarray(self.x, dtype=float)
        self.x2 = self.x[:, range(0, 12)]
        # self.x2 = self.x[:, range(0, 12)]
        self.attributes = np.asarray(df.columns[range(12)])
        self.attribute_units = ['coordinate',
                                'coordinate',
                                'month',
                                'day',
                                'FFMC index',
                                'DMC index',
                                'DC index',
                                'ISI index',
                                '°C',
                                '%',
                                'km/h',
                                'mm/m^2',
                                'ha']

        self.N, self.M = self.x.shape
        self.C = len(self.names)

        # region summary statistics
        self.mean = np.mean(self.x, axis=0)
        self.std = np.std(self.x, axis=0)
        self.min = np.min(self.x, axis=0)
        self.q1 = np.quantile(self.x, 0.25, axis=0)
        self.median = np.median(self.x, axis=0)
        self.q3 = np.quantile(self.x, 0.75, axis=0)
        self.max = np.max(self.x, axis=0)
        self.range = np.max(self.x, axis=0) - np.min(self.x, axis=0)
        # endregion
        
        self.x_tilda = self.x - np.ones((self.N, 1)) * self.mean
        self.x_tilda = self.x_tilda * (1 / np.std(self.x_tilda, axis=0)) 
        
        
        
        # read as pandas dataframe: for plotting
        self.df = pd.read_csv(self.__filename)
        self.df['month'] = df['month'].replace(self.month_dict)
        self.df['day'] = df['day'].replace(self.week_days_dict)
        # self.df['month'] = df['month'].replace(self.month_enc)
        # self.df['day'] = df['day'].replace(self.week_days_enc)
        
        
        data = self.df['month']
        values = np.array(data)
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        
                
        jan, feb,mar,apr,may,jun,jul,aug,sep,otc,nov,dec = [],[],[],[],[],[],[],[],[],[],[],[]
        for vec in onehot_encoded:
            jan.append(vec[0])
            feb.append(vec[1])
            mar.append(vec[2])
            apr.append(vec[3])
            may.append(vec[4])
            jun.append(vec[5])
            jul.append(vec[6])
            aug.append(vec[7])
            sep.append(vec[8])
            otc.append(vec[9])
            nov.append(vec[10])
            dec.append(vec[11])
        self.df['jan'] = jan
        self.df['feb'] = feb
        self.df['mar'] = mar
        self.df['apr'] = apr
        self.df['may'] = may
        self.df['jun'] = jun
        self.df['jul'] = jul
        self.df['aug'] = aug
        self.df['sep'] = sep
        self.df['oct'] = otc
        self.df['nov'] = nov
        self.df['dec'] = dec
        
        
        data = self.df['day']
        values = np.array(data)
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        mon, tue, wed, thu, fri, sat, sun = [],[],[],[],[],[],[]
        for vec in onehot_encoded:
            mon.append(vec[0])
            tue.append(vec[1])
            wed.append(vec[2])
            thu.append(vec[3])
            fri.append(vec[4])
            sat.append(vec[5])
            sun.append(vec[6])
        self.df['mon'] = mon
        self.df['tue'] = tue
        self.df['wed'] = wed
        self.df['thu'] = thu
        self.df['fri'] = fri
        self.df['sat'] = sat
        self.df['sun'] = sun
 
        x1,x2,x3,x4,x5,x6,x7,x8,x9 = [],[],[],[],[],[],[],[],[]

        data = self.df['X']
        values = np.array(data)
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        for vec in onehot_encoded:
            x1.append(vec[0])
            x2.append(vec[1])
            x3.append(vec[2])
            x4.append(vec[3])
            x5.append(vec[4])
            x6.append(vec[5])
            x7.append(vec[6])
            x8.append(vec[7])
            x9.append(vec[8])
        self.df['X1'] = x1
        self.df['X2'] = x2
        self.df['X3'] = x3
        self.df['X4'] = x4
        self.df['X5'] = x5
        self.df['X6'] = x6
        self.df['X7'] = x7
        self.df['X8'] = x8
        self.df['X9'] = x9
        
        
        y1,y2,y3,y4,y5,y6,y7,y8,y9 = [],[],[],[],[],[],[],[],[]
        data = self.df['Y']
        values = np.array(data)
        arr = np.array([1,2,3,4,5,6,7,8,9])
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

        
        for vec in onehot_encoded:
            y1.append(0.0)
            y2.append(vec[0])
            y3.append(vec[1])
            y4.append(vec[2])
            y5.append(vec[3])
            y6.append(vec[4])
            y7.append(0.0)
            y8.append(0.0)
            y9.append(vec[5])
        self.df['Y1'] = y1
        self.df['Y2'] = y2
        self.df['Y3'] = y3
        self.df['Y4'] = y4
        self.df['Y5'] = y5
        self.df['Y6'] = y6
        self.df['Y7'] = y7
        self.df['Y8'] = y8
        self.df['Y9'] = y9
        
        
        # self.df = self.df.iloc[:,4:]
        self.df = self.df.iloc[:,4:13]
        self.df_attributes = list(self.df.columns)
        self.y1 = self.df['area']

        
        #endregion
        

    def get_column_range(self, col_range: range):
        return self.x[:, col_range]

    def get_columns_sorted(self, col_array: [int]):
        cols = []
        col_array = sorted(col_array, reverse=False)

        for col in col_array:
            cols.append(self.x[:, col])

        return cols

    def get_columns(self, col_array: [int]):
        cols = []

        for col in col_array:
            cols.append(self.x[:, col])

        return cols
