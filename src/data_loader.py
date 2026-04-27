import os
import pandas as pd


data_location = 'data/raw/TEP_Faulty_Training.csv'
processed_data_location = 'data/processed/tep_faulty_training_combined.csv'


if os.path.exists(data_location):


    print("SENSOR.AI" + " " + "DATA LOADER")
##if os.path.exists(data_location):

    data = pd.read_csv(data_location)
    print(f"Shape =  {data.shape}")
    print("null values -- ")
    print(data.isnull().sum().sum())
    print(data.describe())

## cleaning empty cells
    data.dropna(inplace = True)
##for now i am taking a small random data
    final_data= data.sample(frac =0.05 , random_state = 42)

    print(f"small data shape ={final_data.shape}")
    final_data.to_csv(processed_data_location, index=False)

    print("small sample generated")
##data = pd.DataFrame(data , index=False)
##print(data.info())
else:
    print("invalid file path")

