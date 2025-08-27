import pandas as pd
import pickle

def read_and_encode_dataset(dataset_name, endcode_ = True, COLUMNS = ""):
    if dataset_name == "celeba":
        data = pd.read_csv(f'datasets/{dataset_name}.txt', sep='\s+', engine='python', skiprows=1)

    elif dataset_name == "adult":
        data = pd.read_csv(f'datasets/{dataset_name}.csv', engine='python')
        
    elif dataset_name == "cardio":
        data = pd.read_csv(f'datasets/{dataset_name}.csv', engine='python')

    elif dataset_name == "SPM_2016":
         data = pd.read_csv(f'datasets/{dataset_name}.csv', engine='python')

    elif dataset_name == "dss":
         data = pd.read_csv(f'datasets/{dataset_name}.csv', engine='python')
    
    else:
        raise ValueError(f"Unkown dataset {dataset_name}")
    
    with open(f'dataset_preprocess_metadata/{dataset_name}/{dataset_name}_metadata.pkl', 'rb') as file:
            load_data = pickle.load(file)
            encoders = load_data["encoders"]
            columns = load_data["columns"]

    if COLUMNS != "":
        columns = COLUMNS

    data = data.replace('?', pd.NA).dropna()
    data.dropna(inplace=True)
    
    if endcode_:
        for col in columns:
            le = encoders[col]
            data[col] = le.fit_transform(data[col])

    return data, columns
