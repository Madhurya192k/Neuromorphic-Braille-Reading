import os
import pickle
import numpy as np
import torch

# Function to load data and labels
def load_data_and_labels(folder_path):
    labels_dict = {'on': 1, 'off': 0}
    data_list, label_list, file_names = [], [], []

    print("Scanning folder for event files...")
    files_in_folder = os.listdir(folder_path)
    print(f"Total files found: {len(files_in_folder)}")

    for file_name in files_in_folder:
        if 'events' in file_name.lower():
            file_path = os.path.join(folder_path, file_name)
            label = next((value for key, value in labels_dict.items() if key in file_name.lower()), None)
            if label is not None:
                label_list.append(label)
                file_names.append(file_name)
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    data_list.append(data)
                except Exception as e:
                    print(f"Failed to load data from {file_name}: {e}")
            else:
                print(f"No matching label found for {file_name}")
    print('Data loading complete.')
    return data_list, label_list, file_names

# Function to convert data to events
def convert_to_events(data_list):
    all_events = []
    for data_item in data_list:
        X, H = data_item.shape[:2]
        events = [(x, y, t, 1) for x in range(X) for y in range(H) for t in data_item[x, y]]
        sorted_events = sorted(events, key=lambda e: e[2])
        all_events.append(sorted_events)
    return all_events

# Function to convert events to frames
def events_to_frames(higher_dim_events, frames_number, P, H, W):
    all_frames = []
    for events in higher_dim_events:
        N = len(events)
        frame_size = max(N // frames_number, 1)
        frames = np.zeros((frames_number, P, H, W), dtype=np.int32)
        for event_index, event in enumerate(events):
            if len(event) != 4:
                print(f"Event does not match expected format (x, y, t, p): {event}")
                continue
            x, y, t, p = event
            if not (0 <= x < H and 0 <= y < W):
                print(f"Skipping out-of-bounds event: x={x}, y={y}, t={t}, p={p}")
                continue
            frame_index = min(event_index // frame_size, frames_number - 1)
            frames[frame_index, 0, x, y] += 1
        all_frames.append(frames)
    return all_frames

# Function to save data with pickle
def save_data_with_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

# Function to load data with pickle
def load_data_with_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Main workflow
if __name__ == "__main__":
    path_to_data = r"C:\Users\madhu\Desktop\test_data\events"
    np.set_printoptions(threshold=np.inf)

    data, label, file = load_data_and_labels(path_to_data)

    if len(file) > 0:
        print("Labels count:", len(label))
        print("File names count:", len(file))
        print("Example filename:", file[0])
    else:
        print("No valid event files were found.")
        exit()

    metadata_save_path = r'C:\Users\madhu\Desktop\test_data\eventdata.pkl'
    labels_save_path = r'C:\Users\madhu\Desktop\test_data\labels.pkl'
    file_save_path = r'C:\Users\madhu\Desktop\test_data\filename.pkl'
    
    save_data_with_pickle(data, metadata_save_path)
    print("metadata saved")
    save_data_with_pickle(label, labels_save_path)
    print("labels saved")
    save_data_with_pickle(file, file_save_path)
    print("filename saved")
    
    # Load saved data
    loaded_data = load_data_with_pickle(metadata_save_path)
    loaded_labels = load_data_with_pickle(labels_save_path)
    loaded_filename = load_data_with_pickle(file_save_path)
    
    data1 = np.array(loaded_data)
    labels1 = np.array(loaded_labels)
    filename1 = np.array(loaded_filename)
    
    print("metadata shape:", data1.shape)
    print("labels shape:", labels1.shape)
    print("filename shape:", filename1.shape)
    
    events = convert_to_events(data1)
    file_path = r'C:\Users\madhu\Desktop\test_data\eventstream_list.pkl'
    save_data_with_pickle(events, file_path)
    loaded_events = load_data_with_pickle(file_path)
    print("Number of event streams:", len(loaded_events))
    
    H, W = 480, 640
    P = 1
    frames_number = 15
    frames = events_to_frames(loaded_events, frames_number, P, H, W)
    
    # ... (Previous code for loading and processing data)

    # Assuming 'frames' and 'loaded_labels' are already available from the previous processing
    all_frames_tensor = torch.stack([torch.tensor(frame, dtype=torch.float32) for frame in frames])
    labels_tensor = torch.tensor(loaded_labels, dtype=torch.long)

    # Define frame data and label save paths (Make sure these paths are correct)
    frames_save_path = r'C:\Users\madhu\Desktop\test_data\data_frames.pt'
    labels_save_path = r'C:\Users\madhu\Desktop\test_data\data_labels.pt'

    torch.save(all_frames_tensor, frames_save_path)
    torch.save(labels_tensor, labels_save_path)
    print("Tensors saved successfully.")
    print (all_frames_tensor)

# Assuming these are the tensors you are saving
# all_frames_tensor = ...
# labels_tensor = ...

print(f'all_frames_tensor type: {type(all_frames_tensor)}')
print(f'all_frames_tensor shape: {all_frames_tensor.shape if isinstance(all_frames_tensor, torch.Tensor) else "Not a Tensor"}')
print(f'all_frames_tensor dtype: {all_frames_tensor.dtype if isinstance(all_frames_tensor, torch.Tensor) else "Not a Tensor"}')

print(f'labels_tensor type: {type(labels_tensor)}')
print(f'labels_tensor shape: {labels_tensor.shape if isinstance(labels_tensor, torch.Tensor) else "Not a Tensor"}')
print(f'labels_tensor dtype: {labels_tensor.dtype if isinstance(labels_tensor, torch.Tensor) else "Not a Tensor"}')

# Save tensors
torch.save(all_frames_tensor, 'data_frames.pt')
torch.save(labels_tensor, 'data_labels.pt')

