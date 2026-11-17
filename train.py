import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from tqdm import tqdm

from data import get_data_loaders, info
from model import DBSSM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device:{device}')

num_rounds = 10
num_epochs = 100
num_classes = info["num_classes"]

oa_list, aa_list, kappa_list = [], [], []
class_acc_all_rounds = [[] for _ in range(num_classes)]

for round_idx in range(num_rounds):
    print(f"\n========== Round {round_idx + 1}/{num_rounds} ==========")

    train_loader, test_loader = get_data_loaders()

    model = DBSSM(spe_channels=info['num_bands'],spa_channels=info['pca_components'],num_classes=info["num_classes"],).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct_train = 0
        total_train = 0

        for patch, spectrum, labels in tqdm(train_loader,
                                            desc=f'Round {round_idx + 1} Epoch [{epoch + 1}/{num_epochs}]'):
            patch = patch.to(device, non_blocking=True)
            spectrum = spectrum.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(patch, spectrum)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

    model.eval()
    y_true = []
    y_pred_test = []
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for patch, spectrum, labels in tqdm(test_loader, desc='Testing'):
            patch = patch.to(device, non_blocking=True)
            spectrum = spectrum.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(patch, spectrum)
            outputs = torch.argmax(outputs, dim=1).cpu().numpy()

            y_true.extend(labels.cpu().numpy())
            y_pred_test.extend(outputs)

            total_test += labels.size(0)
            correct_test += (outputs == labels.cpu().numpy()).sum().item()

    conf_matrix = confusion_matrix(y_true, y_pred_test)
    class_accuracy = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    AA = np.mean(class_accuracy) * 100
    OA = 100 * correct_test / total_test
    kappa = cohen_kappa_score(y_true, y_pred_test) * 100

    aa_list.append(AA)
    oa_list.append(OA)
    kappa_list.append(kappa)

    for i, acc in enumerate(class_accuracy):
        class_acc_all_rounds[i].append(acc * 100)
        print(f'Class {i + 1} accuracy: {acc * 100:.2f}%')

    print(f"Round {round_idx + 1}  OA: {OA:.4f}% |  AA: {AA:.4f}% |  Kappa: {kappa:.4f}%")

print("\n========== Final Results Across Rounds ==========")
print(f"OA Mean: {np.mean(oa_list):.2f} ± {np.std(oa_list):.2f}")
print(f"AA Mean: {np.mean(aa_list):.2f} ± {np.std(aa_list):.2f}")
print(f"Kappa Mean: {np.mean(kappa_list):.2f} ± {np.std(kappa_list):.2f}")

print("\n==== Per-Class Accuracy (Mean ± Std over 10 Rounds) ====")
for i in range(num_classes):
    mean_acc = np.mean(class_acc_all_rounds[i])
    std_acc = np.std(class_acc_all_rounds[i])
    print(f'Class {i + 1} accuracy: {mean_acc:.2f} ± {std_acc:.2f}')