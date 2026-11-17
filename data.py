import numpy as np
import scipy.io as sio
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

dataset_name = "IP"

dataset_info = {
    "IP": {
        "data_name": "Indian Pines",
        "data_file": "datasets/Indian_pines/Indian_pines_corrected.mat",
        "label_file": "datasets/Indian_pines/Indian_pines_gt.mat",
        "data_key": "indian_pines_corrected",
        "label_key": "indian_pines_gt",
        "pca_components": 30,
        "patch_size": 9,
        "batch_size": 64,
        "num_classes": 16,
        "num_bands": 200,
        "test_size": 0.9,
    }
}

info = dataset_info[dataset_name]

X = sio.loadmat(info["data_file"])[info["data_key"]]  # [H, W, C]
y = sio.loadmat(info["label_file"])[info["label_key"]]  # [H, W]

def extract_samples(original_data, labels, patch_size):
    h, w, c = original_data.shape
    margin = (patch_size - 1) // 2
    valid_pixels = []
    for i in range(h):
        for j in range(w):
            label = labels[i, j]
            if label == 0:
                continue
            valid_pixels.append((i, j, label - 1))
    return valid_pixels

valid_pixels = extract_samples(X, y, info["patch_size"])

coords = np.array([[i, j] for (i, j, _) in valid_pixels])
labels = np.array([label for (_, _, label) in valid_pixels])

coords_train, coords_test, y_train, y_test = train_test_split(
    coords, labels,
    test_size=info["test_size"],
    stratify=labels,
    random_state=42
)

h, w, c = X.shape

reshaped = X.reshape(-1, c)

train_indices = coords_train[:, 0] * w + coords_train[:, 1]
train_spectra = reshaped[train_indices]

pca = PCA(n_components=info["pca_components"])
pca.fit(train_spectra)

X_pca = pca.transform(reshaped).reshape(h, w, info["pca_components"])

def build_dataset_from_coords(coords, original_data, pca_data, labels, patch_size):
    margin = (patch_size - 1) // 2
    # data_padded = np.pad(pca_data, ((margin, margin), (margin, margin), (0, 0)), mode='constant')
    data_padded = np.pad(pca_data, ((margin, margin), (margin, margin), (0, 0)), mode='reflect')

    patches, spectra, patch_labels = [], [], []
    for (i, j), label in zip(coords, labels):
        patch = data_padded[i:i + patch_size, j:j + patch_size, :]
        spectrum = original_data[i, j, :]
        patches.append(patch)
        spectra.append(spectrum)
        patch_labels.append(label)
    return np.array(patches), np.array(spectra), np.array(patch_labels)

X_patch_train, X_spec_train, y_train = build_dataset_from_coords(coords_train, X, X_pca, y_train, info["patch_size"])
X_patch_test, X_spec_test, y_test = build_dataset_from_coords(coords_test, X, X_pca, y_test, info["patch_size"])


class HyperspectralDataset(Dataset):
    def __init__(self, patches, spectra, labels):
        self.patches = torch.from_numpy(patches).float().permute(0, 3, 1, 2)
        self.spectra = torch.from_numpy(spectra).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.patches[idx], self.spectra[idx], self.labels[idx]


def get_data_loaders(batch_size=info['batch_size']):
    train_dataset = HyperspectralDataset(X_patch_train, X_spec_train, y_train)
    test_dataset = HyperspectralDataset(X_patch_test, X_spec_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    print(f"Dataset: {info['data_name']}")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Patch shape: {X_patch_train.shape[1:]}, Spectrum dim: {X_spec_train.shape[1]}")

    return train_loader, test_loader