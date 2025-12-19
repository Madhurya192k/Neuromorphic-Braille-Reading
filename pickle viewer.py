import pickle

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def inspect_data(data):
    print("Type of data:", type(data))
    if isinstance(data, dict):
        print("Keys:", data.keys())
    elif isinstance(data, list):
        print("Length of list:", len(data))
        print("First element:", data[0])
    elif isinstance(data, np.ndarray):
        print("Shape of array:", data.shape)
    else:
        print("Data:", data)

# Use the raw string or double backslashes in the file path
file_path = r'C:\Users\madhu\Desktop\horizontal_shear_07171519\labels.pkl'

data = load_pickle(file_path)
inspect_data(data)
