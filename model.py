import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import recall_score, confusion_matrix, classification_report
from collections import Counter
import io
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

USE_RGB_IMAGES = True
EPOCHS = 100
K_FOLDS = 3
BATCH_SIZE = 64
IMAGE_SIZE = (128, 128)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

dataset_path = "dinosaur_dataset"
LOG_DIR_BASE = "runs/dinosaur_experiments_v8"

def plot_to_image(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    return image

def visualize_class_distribution(samples, class_names, writer, log_tag_prefix):
    labels = [s[1] for s in samples]
    class_counts = Counter(labels)
    sorted_class_counts = {name: class_counts.get(name, 0) for name in class_names}

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(sorted_class_counts.keys(), sorted_class_counts.values())
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    ax.set_title("Class Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    os.makedirs("./plots", exist_ok=True)
    plot_path = f"./plots/class_distribution.png"
    fig.savefig(plot_path)
    print(f"Class distribution plot saved to {plot_path}")

    img_tensor = transforms.ToTensor()(plot_to_image(fig))
    writer.add_image(f'{log_tag_prefix}/Class_Distribution', img_tensor, 0)
    plt.close(fig)

def visualize_pixel_histograms(samples_tuples, class_names, writer, log_tag_prefix):
    if not samples_tuples:
        print("No samples provided for pixel histogram visualization.")
        return

    class_images = {class_name: [] for class_name in class_names}
    for img_path, label_str in samples_tuples:
        if label_str in class_images:
            class_images[label_str].append(img_path)

    for class_name, img_paths in class_images.items():
        if not img_paths:
            print(f"No images found for class {class_name} for pixel histogram.")
            continue

        aggr_cnt_r = np.zeros(256, dtype=np.int64)
        aggr_cnt_g = np.zeros(256, dtype=np.int64)
        aggr_cnt_b = np.zeros(256, dtype=np.int64)
        processed_img_cnt = 0

        print(f"Processing pixel histograms for class: {class_name} ({len(img_paths)} images)...")
        for img_path in img_paths:
            try:
                img_pil = Image.open(img_path).convert('RGB')
                img_np = np.array(img_pil)

                r_hist, _ = np.histogram(img_np[:, :, 0].ravel(), bins=256, range=(0, 255))
                g_hist, _ = np.histogram(img_np[:, :, 1].ravel(), bins=256, range=(0, 255))
                b_hist, _ = np.histogram(img_np[:, :, 2].ravel(), bins=256, range=(0, 255))

                aggr_cnt_r += r_hist
                aggr_cnt_g += g_hist
                aggr_cnt_b += b_hist
                processed_img_cnt +=1
            except Exception as e:
                print(f"Skipping image {img_path} for histogram due to error: {e}")
                continue

        if processed_img_cnt == 0:
            continue

        channel_data_map = {
            "Red": aggr_cnt_r,
            "Green": aggr_cnt_g,
            "Blue": aggr_cnt_b
        }
        colors = {'Red': 'red', 'Green': 'green', 'Blue': 'blue'}
        bin_centers = np.arange(256)

        for channel_name, channel_pixel_cnt in channel_data_map.items():
            if np.sum(channel_pixel_cnt) == 0:
                print(f"No {channel_name} pixel data for class {class_name} after processing. Skipping histogram.")
                continue

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(bin_centers, channel_pixel_cnt, width=1.0, color=colors[channel_name], alpha=0.7)
            ax.set_title(f'Aggregated {channel_name} Channel Pixel Intensity\nClass: {class_name}')
            ax.set_xlabel('Pixel Intensity (0-255)')
            ax.set_ylabel('Frequency (across all images in class)')
            ax.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()

            img_tensor = transforms.ToTensor()(plot_to_image(fig))
            writer.add_image(f'{log_tag_prefix}/Pixel_Histograms/{class_name}/{channel_name}_Channel', img_tensor, 0)
            plt.close(fig)
            print(f"Aggregated {channel_name} pixel histogram for class {class_name} logged to TensorBoard.")

def get_image_label_pairs(dataset_path):
    samples = []
    class_names = []
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
    for class_label_str in sorted(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_label_str)
        if os.path.isdir(class_path):
            class_names.append(class_label_str)
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for filename in image_files:
                img_path = os.path.join(class_path, filename)
                samples.append((img_path, class_label_str))
    if not samples:
        raise ValueError(f"No image samples found in {dataset_path}.")
    return samples, class_names

class DinosaurDataset(Dataset):
    def __init__(self, samples, label_encoder, image_size=(128, 128), use_rgb=True, is_train=False):
        self.samples = samples
        self.image_size = image_size
        self.label_encoder = label_encoder
        self.use_rgb = use_rgb
        self.is_train = is_train

        base_transform_list = [
            transforms.Resize(self.image_size),
        ]

        if self.is_train:
            augmentation_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
            ]
            base_transform_list.extend(augmentation_transforms)

        if use_rgb:
            normalization_transforms = [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        else:
            normalization_transforms = [
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ]
        base_transform_list.extend(normalization_transforms)
        self.transform = transforms.Compose(base_transform_list)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_str = self.samples[idx]
        try:
            img = Image.open(img_path)
            if self.use_rgb:
                img = img.convert('RGB')
            else:
                img = img.convert('L')
            img = self.transform(img)
        except IOError as e:
            print(f"Error opening image {img_path}: {e}")
            num_channels = 3 if self.use_rgb else 1
            img = torch.zeros((num_channels, self.image_size[0], self.image_size[1]))
            label_idx = 0
            return img, torch.tensor(label_idx, dtype=torch.long)

        label = self.label_encoder.transform([label_str])[0]
        return img, torch.tensor(label, dtype=torch.long)

all_samples_tuples, class_names = get_image_label_pairs(dataset_path)
if not all_samples_tuples or not class_names:
    raise ValueError("No samples or classes found.")

NUM_CLASSES = len(class_names)
label_encoder = LabelEncoder()
label_encoder.fit(class_names)

string_labels_for_split = [s[1] for s in all_samples_tuples]
train_samples_all, val_samples_all = train_test_split(
    all_samples_tuples,
    test_size=0.25,
    random_state=32,
    stratify=string_labels_for_split if len(string_labels_for_split) > 0 and NUM_CLASSES > 1 else None
)


class DinosaurCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, image_h=IMAGE_SIZE[0], image_w=IMAGE_SIZE[1], use_rgb=True):
        super(DinosaurCNN, self).__init__()
        input_channels = 3 if use_rgb else 1
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flattened_h = image_h // 8
        self.flattened_w = image_w // 8
        self.flattened_size = 64 * self.flattened_h * self.flattened_w

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = x.view(-1, self.flattened_size)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return self.log_softmax(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, current_experiment_config, writer, label_encoder_classes, num_classes_for_cm, device, scheduler=None):
    log_tag_prefix = current_experiment_config['log_tag']
    best_val_loss = float('inf')
    best_val_acc_for_summary = 0.0

    model_save_dir = os.path.join(LOG_DIR_BASE, log_tag_prefix, "models")
    os.makedirs(model_save_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss, total_samples, correct_predictions = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            if isinstance(criterion, nn.KLDivLoss):
                target_probs = F.one_hot(labels, num_classes=num_classes_for_cm).float()
                loss = criterion(outputs, target_probs)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / total_samples if total_samples > 0 else 0
        epoch_train_acc = 100.0 * correct_predictions / total_samples if total_samples > 0 else 0
        writer.add_scalar(f'{log_tag_prefix}/Loss/Train', epoch_train_loss, epoch)
        writer.add_scalar(f'{log_tag_prefix}/Accuracy/Train', epoch_train_acc, epoch)
        print(f"Run: {log_tag_prefix} - Epoch [{epoch+1}/{epochs}] Train Loss: {epoch_train_loss:.4f} Train Acc: {epoch_train_acc:.2f}%")

        model.eval()
        val_running_loss, val_total_samples, val_correct_predictions = 0.0, 0, 0
        all_val_labels = []
        all_val_preds = []
        total_val_entropy = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                probs = torch.exp(outputs)
                batch_entropy = torch.special.entr(probs).sum(dim=1)
                total_val_entropy += batch_entropy.sum().item()

                if isinstance(criterion, nn.KLDivLoss):
                    target_probs = F.one_hot(labels, num_classes=num_classes_for_cm).float()
                    loss = criterion(outputs, target_probs)
                else:
                    loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total_samples += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())

        epoch_val_loss = val_running_loss / val_total_samples if val_total_samples > 0 else 0
        epoch_val_acc = 100.0 * val_correct_predictions / val_total_samples if val_total_samples > 0 else 0
        avg_epoch_val_entropy = total_val_entropy / val_total_samples if val_total_samples > 0 else 0

        if val_total_samples > 0:
            val_recall_macro = recall_score(all_val_labels, all_val_preds, average='macro', zero_division=0)
            writer.add_scalar(f'{log_tag_prefix}/Recall/Val_Macro', val_recall_macro * 100.0, epoch)
            print(f"Run: {log_tag_prefix} - Epoch [{epoch+1}/{epochs}] Val Recall (Macro): {val_recall_macro*100.0:.2f}%")
            writer.add_scalar(f'{log_tag_prefix}/Certainty/Val_Avg_Entropy', avg_epoch_val_entropy, epoch)

        writer.add_scalar(f'{log_tag_prefix}/Loss/Val', epoch_val_loss, epoch)
        writer.add_scalar(f'{log_tag_prefix}/Accuracy/Val', epoch_val_acc, epoch)
        current_lr_for_log = optimizer.param_groups[0]['lr']
        writer.add_scalar(f'{log_tag_prefix}/LearningRate', current_lr_for_log, epoch)
        print(f"Run: {log_tag_prefix} - Epoch [{epoch+1}/{epochs}] Val Loss: {epoch_val_loss:.4f} Val Acc: {epoch_val_acc:.2f}% LR: {current_lr_for_log:.6f}")

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

        if epoch_val_loss < best_val_loss and val_total_samples > 0:
            best_val_loss = epoch_val_loss
            best_val_acc_for_summary = epoch_val_acc
            torch.save(model.state_dict(), os.path.join(model_save_dir, "dino_model_best.pth"))
            print(f"Best model for {log_tag_prefix} saved with val_loss: {best_val_loss:.4f}")

    if val_total_samples > 0:
        cm = confusion_matrix(all_val_labels, all_val_preds, labels=np.arange(num_classes_for_cm))
        fig_cm = plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder_classes, yticklabels=label_encoder_classes, cmap='Blues')
        plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title(f'Confusion Matrix - {log_tag_prefix}')
        plt.tight_layout()
        writer.add_image(f'{log_tag_prefix}/ConfusionMatrix', transforms.ToTensor()(plot_to_image(fig_cm)), epochs -1)
        plt.close(fig_cm)

        report = classification_report(all_val_labels, all_val_preds, target_names=label_encoder_classes, labels=np.arange(num_classes_for_cm), zero_division=0, output_dict=True)
        for class_name_report, metrics_dict in report.items():
            if isinstance(metrics_dict, dict):
                writer.add_scalar(f'{log_tag_prefix}/Precision/Val_Class_{class_name_report}', metrics_dict.get('precision', 0), epochs -1)
                writer.add_scalar(f'{log_tag_prefix}/Recall/Val_Class_{class_name_report}', metrics_dict.get('recall', 0), epochs -1)
                writer.add_scalar(f'{log_tag_prefix}/F1-Score/Val_Class_{class_name_report}', metrics_dict.get('f1-score', 0), epochs -1)
        writer.add_scalar(f'{log_tag_prefix}/F1-Score/Val_Macro', report['macro avg']['f1-score'], epochs -1)
    return {'best_val_loss': best_val_loss, 'best_val_acc': best_val_acc_for_summary}

def run_exps():
    if DEVICE.type == 'mps':
        print("Using MPS (Apple Silicon GPU)")
    elif DEVICE.type == 'cuda':
        print("Using CUDA GPU")
    else:
        print("Using CPU")

    all_samples_tuples, class_names = get_image_label_pairs(dataset_path)
    if not all_samples_tuples or not class_names:
        raise ValueError("No samples or classes found. Please check dataset_path and its contents.")

    NUM_CLASSES = len(class_names)
    print(f"Found {NUM_CLASSES} classes: {class_names}")

    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)

    all_samples_tuples_np = np.array(all_samples_tuples, dtype=object)
    all_string_labels_np = np.array([s[1] for s in all_samples_tuples])

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=12)
    experiment_summary = {}

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(all_samples_tuples_np, all_string_labels_np)):
        print(f"\n===== Starting Fold {fold_idx+1}/{K_FOLDS} =======")

        current_fold_train_samples = all_samples_tuples_np[train_indices].tolist()
        current_fold_val_samples = all_samples_tuples_np[val_indices].tolist()

        experiment_configurations = []
        optimizers_to_test = [
            {"name": "Adam", "class": optim.Adam, "base_params": {}},
            {"name": "SGD_Simple", "class": optim.SGD, "base_params": {}},
            # {"name": "SGD_Momentum", "class": optim.SGD, "base_params": {"momentum": 0.9}},
            {"name": "RMSprop", "class": optim.RMSprop, "base_params": {}}
        ]
        learning_rates_to_test = [1e-3, 1e-4]
        epochs_for_experiments = EPOCHS

        for opt_info in optimizers_to_test:
            for lr_val in learning_rates_to_test:
                current_opt_params = dict(opt_info["base_params"])
                current_opt_params["lr"] = lr_val
                experiment_configurations.append({
                    "batch_size": BATCH_SIZE,
                    "optimizer_details": {"name": opt_info["name"], "class": opt_info["class"], "params": current_opt_params},
                    "scheduler_details": None,
                    "epochs": epochs_for_experiments,
                    "loss_function": {"type": "CrossEntropy"}
                })

        experiment_configurations.append({
            "batch_size": BATCH_SIZE,
            "optimizer_details": {"name": "Adam_StepLR", "class": optim.Adam, "params": {"lr": 1e-3}},
            "scheduler_details": {"class": StepLR, "params": {"step_size": max(1, epochs_for_experiments // 3), "gamma": 0.1}},
            "epochs": epochs_for_experiments,
            "loss_function": {"type": "CrossEntropy"}
        })
        experiment_configurations.append({
            "batch_size": BATCH_SIZE,
            "optimizer_details": {"name": "Adam_KLDiv", "class": optim.Adam, "params": {"lr": 1e-3}},
            "scheduler_details": None,
            "epochs": epochs_for_experiments,
            "loss_function": {"type": "KLDivLoss"}
        })

        if fold_idx == 0:
            initial_writer_for_dist_plot = SummaryWriter(os.path.join(LOG_DIR_BASE, "Dataset_MetaInfo"))
            visualize_class_distribution(all_samples_tuples, class_names, initial_writer_for_dist_plot, "Dataset_MetaInfo")
            visualize_pixel_histograms(all_samples_tuples, class_names, initial_writer_for_dist_plot, "Dataset_MetaInfo")
            initial_writer_for_dist_plot.close()

        for exp_config in experiment_configurations:
            opt_details = exp_config["optimizer_details"]
            opt_name = opt_details["name"]
            opt_class = opt_details["class"]
            opt_params = dict(opt_details["params"])
            current_lr_for_config = opt_params['lr']
            num_epochs_for_run = exp_config.get("epochs", EPOCHS)

            scheduler_details = exp_config.get("scheduler_details")
            scheduler_name_part = f"_Scheduler_{scheduler_details['class'].__name__}" if scheduler_details else ""
            loss_config = exp_config["loss_function"]
            loss_name_part = f"_Loss_{loss_config['type']}"

            base_log_tag = f"Opt_{opt_name}_LR_{current_lr_for_config}_Batch_{BATCH_SIZE}{scheduler_name_part}{loss_name_part}"
            log_tag_with_fold = f"Fold_{fold_idx+1}/{base_log_tag}"

            if base_log_tag not in experiment_summary:
                experiment_summary[base_log_tag] = {'val_losses': [], 'val_accuracies': []}

            writer = SummaryWriter(os.path.join(LOG_DIR_BASE, log_tag_with_fold))

            print(f"\n--- Fold {fold_idx+1}, Experiment: {base_log_tag} for {num_epochs_for_run} epochs ---")

            train_dataset = DinosaurDataset(current_fold_train_samples, label_encoder, image_size=IMAGE_SIZE, use_rgb=USE_RGB_IMAGES, is_train=True)
            val_dataset = DinosaurDataset(current_fold_val_samples, label_encoder, image_size=IMAGE_SIZE, use_rgb=USE_RGB_IMAGES, is_train=False)

            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2, persistent_workers=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=2, persistent_workers=True)

            model = DinosaurCNN(num_classes=NUM_CLASSES, image_h=IMAGE_SIZE[0], image_w=IMAGE_SIZE[1], use_rgb=USE_RGB_IMAGES).to(DEVICE)
            optimizer = opt_class(model.parameters(), **opt_params)

            scheduler = None
            if scheduler_details:
                scheduler = scheduler_details["class"](optimizer, **scheduler_details["params"])

            current_criterion = nn.KLDivLoss(reduction='batchmean') if loss_config["type"] == "KLDivLoss" else nn.CrossEntropyLoss()

            exp_config_for_train_model = exp_config.copy()
            exp_config_for_train_model['log_tag'] = log_tag_with_fold

            results = train_model(model, train_loader, val_loader, current_criterion, optimizer, num_epochs_for_run, exp_config_for_train_model, writer, label_encoder.classes_, NUM_CLASSES, DEVICE, scheduler)

            if 'best_val_loss' in results:
                 experiment_summary[base_log_tag]['val_losses'].append(results['best_val_loss'])
            if 'best_val_acc' in results:
                 experiment_summary[base_log_tag]['val_accuracies'].append(results['best_val_acc'])

            final_model_save_dir = os.path.join(LOG_DIR_BASE, log_tag_with_fold, "models")
            os.makedirs(final_model_save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(final_model_save_dir, "dino_model_final.pth"))
            print(f"Final model for {log_tag_with_fold} saved.")
            writer.close()

    print("\n===== K-Fold Cross-Validation Complete =====")
    print("Summary of best validation results across folds:")
    for config_tag, fold_results in experiment_summary.items():
        if fold_results['val_losses']:
            avg_loss = np.mean(fold_results['val_losses'])
            std_loss = np.std(fold_results['val_losses'])
            print(f"Config: {config_tag} - Avg Best Val Loss: {avg_loss:.4f} +/- {std_loss:.4f}")
        if 'val_accuracies' in fold_results and fold_results['val_accuracies']:
            avg_acc = np.mean(fold_results['val_accuracies'])
            std_acc = np.std(fold_results['val_accuracies'])
            print(f"Config: {config_tag} - Avg Best Val Accuracy: {avg_acc:.2f}% +/- {std_acc:.2f}%")

    print("\nAll experiments complete. View logs with TensorBoard: tensorboard --logdir runs")

if __name__ == '__main__':
    run_exps()
