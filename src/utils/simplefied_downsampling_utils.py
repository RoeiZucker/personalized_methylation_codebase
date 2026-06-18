import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

def replace_labels(indices,downsampled_arr):
    return {
        "labels": [downsampled_arr[i].tolist() for i in indices]
    }
def downsample_dataset(ratio, dataset_to_resample, blank_label,random_state):
    dataset_labels_array = np.array(dataset_to_resample["labels"])

    X, y = convert_dataset_labels_to_position_value_pair(blank_label, dataset_labels_array)
    rus = create_random_sampler(ratio, random_state, y)
    X_resampled, y_resampled = rus.fit_resample(X, y)

    downsampled_arr = create_downsampled_array(blank_label, dataset_labels_array, X_resampled)
    downsampled_train_ds = dataset_to_resample.map(
        lambda x,y:replace_labels(y,downsampled_arr),
        with_indices=True,
        batched=True,
    )
    
    return downsampled_train_ds

def convert_dataset_labels_to_position_value_pair(blank_label, dataset_labels_array):
    mask = dataset_labels_array != blank_label
    example_idx, token_idx = np.where(mask)
    X = np.column_stack([example_idx, token_idx])
    y = dataset_labels_array[example_idx, token_idx].astype(int)
    return X,y

def create_downsampled_array(blank_label, arr, X_resampled):
    downsampled_arr = np.full_like(arr, fill_value=blank_label)
    kept_example_idx = X_resampled[:, 0].astype(int)
    kept_token_idx = X_resampled[:, 1].astype(int)
    downsampled_arr[kept_example_idx, kept_token_idx] = arr[kept_example_idx, kept_token_idx]
    return downsampled_arr

def create_random_sampler(ratio, random_state, y):
    counts = Counter(y)
    minority_count = min(counts.values())
    target_max = int(minority_count / ratio)
    sampling_strategy = {
        class_id: min(class_count, target_max)
        for class_id, class_count in counts.items()
    }

    rus = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
    )
    
    return rus