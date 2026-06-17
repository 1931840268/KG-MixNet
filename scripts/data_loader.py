"""Data loading and augmentation for CoLeaf-DB (GZSL split).

Source domain  : 9 single-deficiency / healthy folders -> seen classes.
Target domain  : the ``more-deficiencies`` folder -> unseen compound classes,
                 with multi-labels parsed from filenames (e.g. ``B_Ca (1).jpg``
                 -> ['B', 'Ca'], ``Ca_He (1).jpg`` -> ['Ca', 'He']).

Set the dataset location via the ``COLEAF_DATA_ROOT`` environment variable, or
pass ``data_root`` to :func:`load_data`. The directory must contain the folders
listed in ``SINGLE_FOLDERS`` plus ``more-deficiencies``.
"""

import os
import glob

from torchvision import transforms

# Default can be overridden by the COLEAF_DATA_ROOT env var or load_data(data_root=...)
DATA_ROOT = os.environ.get("COLEAF_DATA_ROOT", "CoLeaf_raw_images")

SINGLE_FOLDERS = [
    "boron-B", "calcium-Ca", "iron-Fe", "magnesium-Mg",
    "manganese-Mn", "nitrogen-N", "phosphorus-P", "potassium-K", "healthy",
]
COMPOUND_FOLDER = "more-deficiencies"

# Training augmentation (Step-1 physical augmentation in the paper)
train_aug_transforms = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.Resize((224, 224)),               # DINOv3 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Evaluation transform (resize + normalize only)
base_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_data(data_root: str = None):
    """Return ``(source_imgs, source_labels, target_imgs, target_labels)``."""
    root = data_root or DATA_ROOT

    source_imgs, source_labels = [], []
    target_imgs, target_labels = [], []

    # --- source domain (single deficiency + healthy) ---
    for folder in SINGLE_FOLDERS:
        path = os.path.join(root, folder)
        label = folder.split("-")[-1] if "-" in folder else "Healthy"
        for img_path in glob.glob(os.path.join(path, "*.*")):
            source_imgs.append(img_path)
            source_labels.append(label)

    # --- target domain (compound deficiencies) ---
    compound_path = os.path.join(root, COMPOUND_FOLDER)
    for img_path in glob.glob(os.path.join(compound_path, "*.*")):
        filename = os.path.basename(img_path)
        raw_name = filename.split(" ")[0]   # drop the " (1).jpg" suffix
        labels = raw_name.split("_")        # split compound components
        target_imgs.append(img_path)
        target_labels.append(labels)

    print(f"Source (seen) samples : {len(source_imgs)}")
    print(f"Target (unseen) samples: {len(target_imgs)}")
    return source_imgs, source_labels, target_imgs, target_labels
