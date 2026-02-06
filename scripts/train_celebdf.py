import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm
import argparse

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.rheat import DualModel
from src.models.lstm_classifier import LSTMClassifier
from src.data.datasets import CelebDFSeqDataset

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Base Model (EATFormer + ResNet)
    print("Loading Base Model...")
    base_model = DualModel(pretrained_eatformer=False, pretrained_resnet=True).to(device)
    if torch.cuda.device_count() > 1:
        base_model = nn.DataParallel(base_model)
    
    # Load weights if available
    if os.path.exists(args.base_weights):
        print(f"Loading base model weights from {args.base_weights}")
        ckpt = torch.load(args.base_weights, map_location=device)
        # Check if state dict keys match or need adjustment (e.g. module. prefix)
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
        # Handle potential module prefix issue
        # ... logic to strip module. if needed ...
        
        try:
            base_model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Proceeding with partial/random weights.")
    else:
        print(f"Warning: Base model weights not found at {args.base_weights}. Using random init for EATFormer part.")
    
    base_model.eval()

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    # Dataset & Loader - Sequence Extraction
    print("Initializing Dataset for Sequence Extraction...")
    # NOTE: CelebDFSeqDataset expects 'root_dir' to contain 'real' and 'fake' subfolders
    train_ds = CelebDFSeqDataset(
        root_dir=args.data_dir,
        split="Train",
        base_model=base_model,
        transform=transform,
        max_len=args.seq_len
    )
    # Using a smaller batch size for feature extraction essentially
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    print("Extracting sequences (this may take a while)...")
    all_seqs, all_labels = [], []
    # Add error handling for empty dataset
    if len(train_loader) == 0:
        print("Training dataset is empty or paths are incorrect.")
        return

    for seqs, labs in tqdm(train_loader, desc="Extracting features"):
        all_seqs.append(seqs)
        all_labels.append(labs)
    
    if not all_seqs:
        print("No data extracted. Exiting.")
        return

    all_seqs   = torch.cat(all_seqs, dim=0)   # (N, max_len, 1)
    all_labels = torch.cat(all_labels, dim=0) # (N,)

    # LSTM Training
    print("Training LSTM Classifier...")
    lstm_model = LSTMClassifier().to(device)
    optimizer  = optim.Adam(lstm_model.parameters(), lr=args.lr)
    criterion  = nn.BCEWithLogitsLoss()

    lstm_train_dataset = TensorDataset(all_seqs, all_labels)
    lstm_train_loader = DataLoader(lstm_train_dataset, batch_size=args.lstm_batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        lstm_model.train()
        total_loss = 0
        for seqs, labs in lstm_train_loader:
            seqs, labs = seqs.to(device), labs.to(device)
            logits     = lstm_model(seqs)
            loss       = criterion(logits, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * seqs.size(0)
        
        avg_loss = total_loss / len(lstm_train_loader.dataset)
        print(f"Epoch {epoch}/{args.epochs} — Loss: {avg_loss:.4f}")

    # Save Model
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save({"model_state_dict": lstm_model.state_dict()}, args.output_path)
    print(f"Saved LSTM model to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Celeb-DF LSTM")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Celeb-DF Crop/Train directory containing 'real' and 'fake' folders")
    parser.add_argument("--base_weights", type=str, default="weights/celebdf/eatformer_binary_classification.pth", help="Path to base model weights")
    parser.add_argument("--output_path", type=str, default="weights/celebdf/lstm_best.pth", help="Output path for LSTM weights")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for feature extraction (video level)")
    parser.add_argument("--lstm_batch_size", type=int, default=32, help="Batch size for LSTM training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of LSTM training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seq_len", type=int, default=40, help="Sequence length")

    args = parser.parse_args()
    train(args)
