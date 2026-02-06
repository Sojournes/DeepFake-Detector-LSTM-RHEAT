import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import json
import argparse
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.rheat import DualModel
from src.models.lstm_classifier import LSTMClassifier
from src.data.datasets import FFppFramePredictionsDataset

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Base Model
    print("Loading Base Model...")
    base_model = DualModel(pretrained_eatformer=False, pretrained_resnet=False).to(device)
    if torch.cuda.device_count() > 1:
        base_model = nn.DataParallel(base_model)
        
    if os.path.exists(args.base_weights):
        print(f"Loading base model weights from {args.base_weights}")
        ckpt = torch.load(args.base_weights, map_location=device)
        if "model_state_dict" in ckpt:
            base_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
             base_model.load_state_dict(ckpt, strict=False)
    else:
        print(f"Error: Base weights not found at {args.base_weights}")
        return

    base_model.eval()

    # LSTM Model
    print("Loading LSTM Model...")
    lstm_model = LSTMClassifier().to(device)
    if os.path.exists(args.lstm_weights):
        print(f"Loading LSTM weights from {args.lstm_weights}")
        ckpt = torch.load(args.lstm_weights, map_location=device)
        if "model_state_dict" in ckpt:
            lstm_model.load_state_dict(ckpt["model_state_dict"])
        else:
            lstm_model.load_state_dict(ckpt)
    else:
        print(f"Error: LSTM weights not found at {args.lstm_weights}")
        return
    lstm_model.eval()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    categories = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"] 
    # Adjust categories based on what's available or requested
    
    results = {}

    for category in categories:
        print(f"Testing Category: {category}")
        # this dataset needs a json file with splits, like splits.json
        if not os.path.exists(args.splits_json):
             print(f"Error: Splits JSON not found at {args.splits_json}")
             break
             
        try:
            test_ds = FFppFramePredictionsDataset(
                json_path=args.splits_json,
                root_dir=args.data_dir,
                split="test",
                category=category,
                base_model=base_model,
                transform=transform
            )
        except KeyError:
            print(f"Skipping {category} (not found in JSON)")
            continue
            
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        
        preds, labs = [], []
        if len(test_loader) == 0:
            print(f"No data for {category}")
            continue

        with torch.no_grad():
            for seqs, lb in tqdm(test_loader, desc=f"Testing {category}"):
                seqs = seqs.to(device)
                out  = lstm_model(seqs)
                pr   = torch.sigmoid(out).cpu().tolist()
                preds.extend(pr)
                labs.extend(lb.tolist())
        
        if not preds:
            print(f"No predictions for {category}")
            continue

        try:
            auc  = roc_auc_score(labs, preds)
            binp = [1 if p>0.5 else 0 for p in preds]
            f1   = f1_score(labs, binp)
            acc  = accuracy_score(labs, binp)
            
            print(f"{category} -> AUC: {auc:.4f}, F1: {f1:.4f}, Acc: {acc:.4f}")
            results[category] = {"auc": auc, "f1": f1, "acc": acc}
        except Exception as e:
            print(f"Error calculating metrics for {category}: {e}")

    # Save results
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RHEAT-LSTM on FF++")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory of FF++ data")
    parser.add_argument("--splits_json", type=str, required=True, help="Path to splits JSON file")
    parser.add_argument("--base_weights", type=str, default="weights/ffpp/rheat_best.pth", help="Path to base model weights")
    parser.add_argument("--lstm_weights", type=str, default="weights/ffpp/lstm_best.pth", help="Path to LSTM weights")
    parser.add_argument("--output_json", type=str, default="test_results.json", help="Path to save results")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")

    args = parser.parse_args()
    test(args)
