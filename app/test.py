import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data_full = pd.read_csv("data/full_train.csv", header=0, dtype={'Age': np.float64})

print(f"train full: {data_full}")

for col in data_full.columns:
    print(col)

print("____________________________________")

data_new = data_full.loc[:,
           ["MSSubClass", "MSZoning", "LotFrontage", "Neighborhood", "HouseStyle", "RoofStyle", "Exterior1st",
            "MasVnrType", "ExterQual", "Foundation", "BsmtFinSF1", "HeatingQC", "2ndFlrSF", "GrLivArea", "BsmtFullBath",
            "KitchenQual", "Fireplaces", "GarageType", "MiscVal", "SaleCondition", "SalePrice"]]

print(data_new)

data_new.to_csv("data/new.csv", index=False)

train, val = train_test_split(data_new, test_size=0.2, random_state=42)

print("_________________________________")
print(train)
print(val)

str_columns = ['Neighborhood',
               'HouseStyle',
               'RoofStyle',
               'Exterior1st',
               'MasVnrType',
               'ExterQual',
               'Foundation',
               'HeatingQC',
               'KitchenQual',
               'GarageType']

train[str_columns] = train.loc[:, str_columns].fillna('None')
val[str_columns] = val.loc[:, str_columns].fillna('None')

train.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)

for col in train.columns:
    print(col)
