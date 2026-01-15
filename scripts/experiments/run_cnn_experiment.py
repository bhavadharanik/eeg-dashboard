"""
CNN Baseline Experiment - Lightweight CNN for EEG Emotion Classification
=========================================================================
Trains a lightweight CNN on EEG spectrograms using LOSO cross-validation.
This serves as the baseline for comparison with CLIP-based approach.

Architecture:
- 3 Conv blocks with BatchNorm (32 -> 64 -> 128 channels)
- Global Average Pooling -> 128-dim features
- MLP Classifier (256 -> 128 -> 2)
"""

import os
import sys

# Change to project root directory
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import json
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, balanced_accuracy_score, roc_auc_score
from datetime import datetime
from pathlib import Path

from src.models.cnn import EEG_CNN

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'cache_path': 'results/spectrogram_cache',
    'results_path': 'results/cnn_baseline_experiment',
    'num_subjects': 5,  # Quick validation with 5 subjects (~30-60 min)
    'batch_size': 32,  # Larger batch for speed
    'epochs': 10,  # Fewer epochs
    'learning_rate': 0.001,
    'early_stopping_patience': 3,
    'random_seed': 42,
}

# Device setup - use MPS (Apple Silicon GPU) for faster training
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_subject_data(cache_path, subject_id):
    """Load spectrogram data for a subject."""
    filepath = os.path.join(cache_path, f's{subject_id:02d}_spectrograms.npz')
    if not os.path.exists(filepath):
        return None, None, None

    data = np.load(filepath)
    # Spectrograms are stored as (N, H, W, C) with values 0-255
    specs = data['spectrograms'].astype(np.float32) / 255.0
    valence = data['valence'].astype(np.int64)
    arousal = data['arousal'].astype(np.int64)
    data.close()
    return specs, valence, arousal


class LazyEEGDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset that loads subjects on demand."""

    def __init__(self, cache_path, subject_ids, label_key='valence'):
        self.cache_path = cache_path
        self.subject_ids = subject_ids
        self.label_key = label_key

        # Build index: list of (subject_id, sample_idx)
        self.index = []
        for subj in subject_ids:
            filepath = os.path.join(cache_path, f's{subj:02d}_spectrograms.npz')
            if os.path.exists(filepath):
                data = np.load(filepath)
                n_samples = len(data['valence'])
                data.close()
                for i in range(n_samples):
                    self.index.append((subj, i))

        # Cache for current subject data
        self._cached_subj = None
        self._cached_data = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        subj, sample_idx = self.index[idx]

        # Load subject data if not cached
        if self._cached_subj != subj:
            filepath = os.path.join(self.cache_path, f's{subj:02d}_spectrograms.npz')
            data = np.load(filepath)
            self._cached_data = {
                'specs': data['spectrograms'].astype(np.float32) / 255.0,
                'valence': data['valence'].astype(np.int64),
                'arousal': data['arousal'].astype(np.int64)
            }
            data.close()
            self._cached_subj = subj

        spec = self._cached_data['specs'][sample_idx]
        label = self._cached_data[self.label_key][sample_idx]

        # Convert (H, W, C) -> (C, H, W)
        spec_tensor = torch.FloatTensor(spec).permute(2, 0, 1)
        return spec_tensor, label


def train_cnn_fold_lazy(train_subjects, test_subj, label_key, device=DEVICE):
    """Train CNN model for one LOSO fold."""

    # Load all data into memory (faster for small number of subjects)
    train_specs_list = []
    train_labels_list = []
    for subj in train_subjects:
        specs, v, a = load_subject_data(CONFIG['cache_path'], subj)
        if specs is not None:
            # Convert (N, H, W, C) -> (N, C, H, W)
            specs = np.transpose(specs, (0, 3, 1, 2))
            train_specs_list.append(specs)
            train_labels_list.append(v if label_key == 'valence' else a)

    train_specs = np.concatenate(train_specs_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    del train_specs_list, train_labels_list
    gc.collect()

    # Load test data
    test_specs, test_v, test_a = load_subject_data(CONFIG['cache_path'], test_subj)
    test_specs = np.transpose(test_specs, (0, 3, 1, 2))
    test_labels = test_v if label_key == 'valence' else test_a

    # Create tensor datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(train_specs),
        torch.LongTensor(train_labels)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(test_specs),
        torch.LongTensor(test_labels)
    )
    del train_specs, test_specs
    gc.collect()

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])

    # Initialize model
    model = EEG_CNN(num_classes=2).to(device)

    # Compute class weights for imbalanced data
    class_counts = np.bincount(train_labels, minlength=2)
    total = len(train_labels)
    weights = torch.FloatTensor([total / (2 * max(c, 1)) for c in class_counts]).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_loss = float('inf')
    patience = 0
    best_state = None

    print(f"  Starting training: {len(train_dataset)} train, {len(test_dataset)} test samples", flush=True)

    # Training loop
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        n_batches = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step(avg_loss)

        # Log epoch progress
        print(f"    Epoch {epoch+1}/{CONFIG['epochs']}: loss={avg_loss:.4f}", flush=True)

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= CONFIG['early_stopping_patience']:
                print(f"    Early stopping at epoch {epoch+1}", flush=True)
                break

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    # Evaluation
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            outputs = model(X)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(y.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
    }

    try:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_probs))
    except:
        metrics['roc_auc'] = 0.5

    # Cleanup
    del model, train_loader, test_loader, train_dataset, test_dataset
    gc.collect()

    return metrics, len(train_labels), len(y_true)


def load_checkpoint(results_path):
    """Load checkpoint if exists."""
    checkpoint_file = results_path / 'checkpoint.json'
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None


def save_checkpoint(results_path, results):
    """Save checkpoint after each subject."""
    checkpoint_file = results_path / 'checkpoint.json'
    with open(checkpoint_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  [Checkpoint saved]", flush=True)


def run_loso_task(task_name, label_key, results_path, full_results, existing_results=None):
    """Run LOSO CV for a single task (valence or arousal) with checkpointing after each subject."""
    print(f"\n{'=' * 60}", flush=True)
    print(f"[{task_name.upper()}] Leave-One-Subject-Out Cross-Validation", flush=True)
    print("=" * 60, flush=True)

    # Load existing results or start fresh
    if existing_results and 'per_subject' in existing_results:
        task_results = existing_results
        completed = set(int(k) for k in task_results['per_subject'].keys())
        print(f"Resuming from checkpoint. Completed: {len(completed)} subjects", flush=True)
    else:
        task_results = {'per_subject': {}}
        completed = set()

    # Get list of available subjects
    available_subjects = []
    for subj in range(1, CONFIG['num_subjects'] + 1):
        filepath = os.path.join(CONFIG['cache_path'], f's{subj:02d}_spectrograms.npz')
        if os.path.exists(filepath):
            available_subjects.append(subj)

    for test_subj in available_subjects:
        # Skip if already completed
        if test_subj in completed:
            print(f"\nSubject {test_subj}/{len(available_subjects)} - SKIPPED (already done)", flush=True)
            continue

        print(f"\nSubject {test_subj}/{len(available_subjects)} (test)", flush=True)

        # Get training subjects (all except test)
        train_subjects = [s for s in available_subjects if s != test_subj]

        # Train using lazy loading
        metrics, n_train, n_test = train_cnn_fold_lazy(train_subjects, test_subj, label_key)

        # Get test class distribution
        _, test_v, test_a = load_subject_data(CONFIG['cache_path'], test_subj)
        test_labels = test_v if label_key == 'valence' else test_a
        pos_count = np.sum(test_labels)

        print(f"  Train: {n_train} samples, Test: {n_test} samples", flush=True)
        print(f"  Test class dist: Pos={pos_count}, Neg={n_test-pos_count}", flush=True)
        print(f"  Results: F1={metrics['f1_score']:.3f}, Acc={metrics['accuracy']:.3f}, BAcc={metrics['balanced_accuracy']:.3f}", flush=True)

        task_results['per_subject'][str(test_subj)] = metrics

        # Save checkpoint after each subject
        full_results[label_key] = task_results
        save_checkpoint(results_path, full_results)

        del test_v, test_a, test_labels
        gc.collect()

    return task_results


def run_experiment():
    print("=" * 60, flush=True)
    print("CNN BASELINE EXPERIMENT", flush=True)
    print("Lightweight CNN for EEG Emotion Classification", flush=True)
    print("=" * 60, flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    set_seed(CONFIG['random_seed'])

    results_path = Path(CONFIG['results_path'])
    results_path.mkdir(parents=True, exist_ok=True)

    # Check if spectrograms exist
    if not os.path.exists(os.path.join(CONFIG['cache_path'], 's01_spectrograms.npz')):
        print("\nError: Spectrogram cache not found!")
        print(f"Please ensure spectrograms are cached in {CONFIG['cache_path']}")
        return

    # Check available subjects
    print("\nChecking available subjects...", flush=True)
    available_subjects = []
    for subj in range(1, CONFIG['num_subjects'] + 1):
        filepath = os.path.join(CONFIG['cache_path'], f's{subj:02d}_spectrograms.npz')
        if os.path.exists(filepath):
            available_subjects.append(subj)
            print(f"  S{subj:02d}: OK", flush=True)
        else:
            print(f"  S{subj:02d}: MISSING", flush=True)

    print(f"\nFound {len(available_subjects)}/{CONFIG['num_subjects']} subjects", flush=True)

    # Load checkpoint if exists
    checkpoint = load_checkpoint(results_path)
    if checkpoint:
        print("\n*** RESUMING FROM CHECKPOINT ***", flush=True)
        results = checkpoint
    else:
        results = {
            'valence': {},
            'arousal': {},
            'config': CONFIG,
            'model': 'CNN (3 conv blocks + MLP)',
            'architecture': {
                'conv_blocks': 3,
                'channels': [32, 64, 128],
                'feature_dim': 128,
                'classifier': [256, 128, 2]
            }
        }

    # Run VALENCE LOSO with checkpointing after each subject
    results['valence'] = run_loso_task('valence', 'valence', results_path, results, results.get('valence'))

    # Run AROUSAL LOSO with checkpointing after each subject
    results['arousal'] = run_loso_task('arousal', 'arousal', results_path, results, results.get('arousal'))

    # Calculate summary statistics
    for task in ['valence', 'arousal']:
        if results[task]['per_subject']:
            m = list(results[task]['per_subject'].values())
            results[task]['summary'] = {
                'mean_f1': float(np.mean([x['f1_score'] for x in m])),
                'std_f1': float(np.std([x['f1_score'] for x in m])),
                'mean_accuracy': float(np.mean([x['accuracy'] for x in m])),
                'std_accuracy': float(np.std([x['accuracy'] for x in m])),
                'mean_precision': float(np.mean([x['precision'] for x in m])),
                'mean_recall': float(np.mean([x['recall'] for x in m])),
                'mean_balanced_accuracy': float(np.mean([x['balanced_accuracy'] for x in m])),
                'mean_roc_auc': float(np.mean([x['roc_auc'] for x in m])),
                'min_f1': float(np.min([x['f1_score'] for x in m])),
                'max_f1': float(np.max([x['f1_score'] for x in m])),
            }

    results['timestamp'] = datetime.now().isoformat()

    # Save results
    with open(results_path / 'cnn_baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60, flush=True)
    print("FINAL RESULTS - CNN BASELINE", flush=True)
    print("=" * 60, flush=True)

    if 'summary' in results['valence']:
        v = results['valence']['summary']
        print(f"VALENCE: F1={v['mean_f1']:.3f}±{v['std_f1']:.3f}, "
              f"Acc={v['mean_accuracy']:.3f}±{v['std_accuracy']:.3f}, "
              f"BAcc={v['mean_balanced_accuracy']:.3f}", flush=True)

    if 'summary' in results['arousal']:
        a = results['arousal']['summary']
        print(f"AROUSAL: F1={a['mean_f1']:.3f}±{a['std_f1']:.3f}, "
              f"Acc={a['mean_accuracy']:.3f}±{a['std_accuracy']:.3f}, "
              f"BAcc={a['mean_balanced_accuracy']:.3f}", flush=True)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Results saved to: {results_path / 'cnn_baseline_results.json'}", flush=True)


if __name__ == '__main__':
    run_experiment()
