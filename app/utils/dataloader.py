import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import LabelEncoder


class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):
        # columns combination

        cols_to_drop = ["MSSubClass", "MSZoning", "SaleCondition"]

        # drop columns

        self.dataset = self.dataset.drop(cols_to_drop, axis=1)

        # # replace
        # self.dataset['LotShape']=self.dataset['LotShape'].replace(['IR1', 'IR2', 'IR3'], 'IR')

        str_cols = ['Neighborhood',
                    'HouseStyle',
                    'RoofStyle',
                    'Exterior1st',
                    'MasVnrType',
                    'ExterQual',
                    'Foundation',
                    'HeatingQC',
                    'KitchenQual',
                    'GarageType']
        self.dataset[str_cols] = self.dataset[str_cols].fillna('None')
        le = LabelEncoder()
        for i in str_cols:
            le.fit(self.dataset[i])
            self.dataset[i] = le.transform(self.dataset[i])

        self.dataset = self.dataset.fillna(0)

        return self.dataset
