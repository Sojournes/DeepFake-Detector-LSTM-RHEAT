import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class FaceForensicsDataset(Dataset):
    def __init__(self, json_path, root_dir, split, transform=None):
        """
        Args:
            json_path (str): Path to the JSON file containing the data split information.
            root_dir (str): Base directory for the dataset.
            split (str): One of the splits, e.g., 'train', 'val', or 'test'.
            transform (callable, optional): A function/transform to apply to the images.
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Dynamically iterate over all available categories
        for category, category_data in self.data.get("FaceForensics++", {}).items():
            try:
                videos = category_data[split]["c23"]
            except KeyError:
                print(f"Warning: Category {category} does not have the split '{split}' or key 'c23'.")
                continue

            for video_id, info in videos.items():
                for frame_path in info.get("frames", []):
                    frame_path = frame_path.replace("\\\\", "/")  # Normalize Windows paths
                    if frame_path.startswith("FaceForensics++/"):
                        frame_path = "/".join(frame_path.split("/")[1:])
                    full_path = os.path.join(self.root_dir, frame_path)
                    
                    if os.path.isfile(full_path):
                        label = 1 if category == "FF-real" else 0
                        self.samples.append((full_path, label))
                    else:
                        pass # Silently skip missing

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float)
        except Exception as e:
            print(f"ERROR loading {img_path}: {e}")
            return torch.zeros(3, 224, 224), torch.tensor(label, dtype=torch.float)

class FFppFramePredictionsDataset(Dataset):
    def __init__(self, json_path, root_dir, split, category, base_model, transform=None):
        with open(json_path, 'r') as f:
            data = json.load(f)["FaceForensics++"][category][split]["c23"]
        self.root_dir = root_dir
        self.transform = transform
        self.base_model = base_model.eval()
        self.device = next(base_model.parameters()).device

        self.samples = []
        for vid_id, info in data.items():
            frames = info["frames"]
            full_paths = []
            for f in frames:
                p = f.replace("\\\\", "/")
                if p.startswith("FaceForensics++/"):
                    p = "/".join(p.split("/")[1:])
                full = os.path.join(root_dir, p)
                if os.path.isfile(full):
                    full_paths.append(full)
            if full_paths:
                label = 1 if category == "FF-real" else 0
                self.samples.append((full_paths, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, label = self.samples[idx]
        probs = []
        with torch.no_grad():
            for p in paths:
                img = Image.open(p).convert("RGB")
                if self.transform: img = self.transform(img)
                img = img.unsqueeze(0).to(self.device)
                out = self.base_model(img)
                # Ensure output is a scalar probability
                if out.dim() > 1 and out.size(-1) > 1:
                     # Assuming class 1 is fake/real based on model
                     # But DualModel outputs logits for binary classification (dim 1)
                     prob = torch.sigmoid(out).item()
                else:
                     prob = torch.sigmoid(out).item()
                probs.append(prob)
        # pad/truncate all sequences to same length, e.g. max_len=50
        seq = torch.tensor(probs, dtype=torch.float)
        max_len = 50
        if seq.size(0) < max_len:
            pad = torch.zeros(max_len - seq.size(0), dtype=torch.float)
            seq = torch.cat([seq, pad])
        else:
            seq = seq[:max_len]
        
        return seq.unsqueeze(-1), torch.tensor(label, dtype=torch.float)

class CelebDFSeqDataset(Dataset):
    def __init__(self, root_dir, split,
                 base_model, transform,
                 max_len=50):
        """
        root_dir/
          real/{video_id}/frame*.jpg
          fake/{video_id}/frame*.jpg
        split is unused here (Train vs Test folder passed separately).
        """
        self.real_dir = os.path.join(root_dir, "real")
        self.fake_dir = os.path.join(root_dir, "fake")

        # build (frame_paths_list, label) for each video
        self.videos = []
        # Check if directories exist
        if os.path.isdir(self.real_dir):
             for vid in os.listdir(self.real_dir):
                vid_dir = os.path.join(self.real_dir, vid)
                if not os.path.isdir(vid_dir): continue
                frames = sorted([
                    os.path.join(vid_dir, f)
                    for f in os.listdir(vid_dir)
                    if f.lower().endswith(('.jpg','.png'))
                ])
                if not frames: continue
                self.videos.append((frames, 1)) # 1 for Real
        
        if os.path.isdir(self.fake_dir):
             for vid in os.listdir(self.fake_dir):
                vid_dir = os.path.join(self.fake_dir, vid)
                if not os.path.isdir(vid_dir): continue
                frames = sorted([
                    os.path.join(vid_dir, f)
                    for f in os.listdir(vid_dir)
                    if f.lower().endswith(('.jpg','.png'))
                ])
                if not frames: continue
                self.videos.append((frames, 0)) # 0 for Fake

        self.base      = base_model.eval()
        self.dev       = next(self.base.parameters()).device
        self.transform = transform
        self.max_len   = max_len

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        frames, label = self.videos[idx]
        L = len(frames)

        # uniformly sample indices up to max_len
        if L > self.max_len:
            idxs = np.linspace(0, L-1, self.max_len, dtype=int)
        else:
            idxs = np.arange(L, dtype=int)

        probs = []
        with torch.no_grad():
            for i in idxs:
                img = Image.open(frames[i]).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                img = img.unsqueeze(0).to(self.dev)
                out = self.base(img)
                probs.append(torch.sigmoid(out).item())

        # pad if needed
        if len(probs) < self.max_len:
            probs += [0.0] * (self.max_len - len(probs))

        # (max_len, 1)
        seq = torch.tensor(probs, dtype=torch.float).unsqueeze(-1)
        return seq, torch.tensor(label, dtype=torch.float)
