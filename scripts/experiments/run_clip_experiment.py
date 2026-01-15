"""
Improved EEG Experiment with ML Fixes
======================================
Implements:
1. SMOTE - Synthetic Minority Oversampling
2. Focal Loss - Better learning on imbalanced data
3. Threshold Optimization - Find optimal decision boundary

Based on 32-subject experiment structure.
"""

import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
import time
import gc
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import signal
from scipy.stats import ttest_rel, wilcoxon
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# SMOTE for oversampling
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    print("Warning: imbalanced-learn not installed. Installing...")
    os.system("pip install imbalanced-learn -q")
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True

# Force output flushing
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_path': 'data/deap',
    'features_path': 'results/32subject_features',  # Pre-extracted features
    'results_path': 'results/improved_experiment',
    'checkpoint_path': 'results/improved_checkpoints',
    'num_subjects': 32,
    'batch_size': 32,
    'clip_epochs': 30,
    'learning_rate': 0.0005,  # Slightly lower for stability
    'early_stopping_patience': 7,
    'random_seed': 42,
    # New parameters for improvements
    'use_smote': True,
    'use_focal_loss': True,
    'use_threshold_optimization': True,
    'focal_gamma': 2.0,  # Focal loss gamma parameter
    'focal_alpha': 0.25,  # Focal loss alpha parameter
    'threshold_candidates': [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
}

# Device setup
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# Create directories
for path in [CONFIG['features_path'], CONFIG['results_path'], CONFIG['checkpoint_path']]:
    os.makedirs(path, exist_ok=True)

np.random.seed(CONFIG['random_seed'])
torch.manual_seed(CONFIG['random_seed'])

print(f"Using device: {DEVICE}")
print(f"Improvements enabled:")
print(f"  - SMOTE: {CONFIG['use_smote']}")
print(f"  - Focal Loss: {CONFIG['use_focal_loss']}")
print(f"  - Threshold Optimization: {CONFIG['use_threshold_optimization']}")


# ============================================================================
# FOCAL LOSS IMPLEMENTATION
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Paper: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    - gamma: focusing parameter (default 2.0)
      - Higher gamma = more focus on hard examples
    - alpha: class weight (default 0.25 for positive class)
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt is the probability of correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            # Apply class weights
            if isinstance(self.alpha, (float, int)):
                alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class FocalLossWithWeights(nn.Module):
    """Focal Loss with automatic class weight computation."""
    def __init__(self, gamma=2.0, class_weights=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute softmax probabilities
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()

        # Get probability of true class
        pt = (probs * targets_one_hot).sum(dim=1)

        # Compute focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ============================================================================
# SMOTE IMPLEMENTATION
# ============================================================================

def apply_smote(features, labels, random_state=42):
    """
    Apply SMOTE to oversample minority class.

    Returns balanced features and labels.
    """
    # Count classes
    class_counts = np.bincount(labels)
    minority_class = np.argmin(class_counts)
    majority_class = np.argmax(class_counts)

    imbalance_ratio = class_counts[majority_class] / max(class_counts[minority_class], 1)

    print(f"    Before SMOTE: Class 0={class_counts[0]}, Class 1={class_counts[1]}, Ratio={imbalance_ratio:.2f}")

    # Only apply if significant imbalance
    if imbalance_ratio > 1.5:
        try:
            smote = SMOTE(
                sampling_strategy='minority',  # Only oversample minority
                random_state=random_state,
                k_neighbors=min(5, class_counts[minority_class] - 1)  # Adjust k for small minority
            )
            features_resampled, labels_resampled = smote.fit_resample(features, labels)

            new_counts = np.bincount(labels_resampled)
            print(f"    After SMOTE: Class 0={new_counts[0]}, Class 1={new_counts[1]}")

            return features_resampled.astype(np.float32), labels_resampled
        except Exception as e:
            print(f"    SMOTE failed: {e}. Using original data.")
            return features, labels
    else:
        print(f"    SMOTE skipped (ratio <= 1.5)")
        return features, labels


# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================

def optimize_threshold(model, val_features, val_labels, thresholds=None):
    """
    Find optimal classification threshold that maximizes F1 score.

    Uses validation set to find the best threshold.
    """
    if thresholds is None:
        thresholds = CONFIG['threshold_candidates']

    model.eval()

    # Get probabilities
    with torch.no_grad():
        features_tensor = torch.FloatTensor(val_features).to(DEVICE)
        outputs = model(features_tensor)
        probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()

    best_threshold = 0.5
    best_f1 = 0.0

    from sklearn.metrics import f1_score

    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(val_labels, preds, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    return best_threshold, best_f1


def predict_with_threshold(model, features, threshold=0.5):
    """Make predictions using optimized threshold."""
    model.eval()
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features).to(DEVICE)
        outputs = model(features_tensor)
        probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()

    predictions = (probs >= threshold).astype(int)
    return predictions, probs


# ============================================================================
# MODELS
# ============================================================================

class ImprovedCLIPClassifier(nn.Module):
    """
    Improved MLP classifier for CLIP features.

    Enhancements:
    - Layer normalization for stability
    - Residual connections
    - Better regularization
    """
    def __init__(self, input_dim=512, hidden_dims=[384, 256, 128]):
        super().__init__()

        self.input_norm = nn.LayerNorm(input_dim)

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),  # GELU often works better than ReLU
                nn.Dropout(0.3 if i < len(hidden_dims) - 1 else 0.2)
            ])
            prev_dim = hidden_dim

        self.hidden = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dims[-1], 2)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.hidden(x)
        return self.classifier(x)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_subject_features(subj_id):
    """Load pre-extracted features from disk."""
    feature_file = Path(CONFIG['features_path']) / f"s{subj_id:02d}_features.npz"
    if not feature_file.exists():
        return None, None, None
    data = np.load(feature_file)
    return data['clip_features'], data['valence_labels'], data['arousal_labels']


# ============================================================================
# IMPROVED TRAINING
# ============================================================================

def train_improved_model(train_features, train_labels, test_features, test_labels,
                         use_smote=True, use_focal_loss=True, use_threshold_opt=True):
    """
    Train CLIP classifier with all improvements.

    Steps:
    1. Apply SMOTE to training data
    2. Split training into train/val for threshold optimization
    3. Train with Focal Loss
    4. Optimize threshold on validation set
    5. Evaluate on test set
    """

    # Step 1: Apply SMOTE if enabled
    if use_smote:
        train_features, train_labels = apply_smote(train_features, train_labels)

    # Step 2: Split for threshold optimization (90% train, 10% val)
    if use_threshold_opt:
        n_train = len(train_labels)
        indices = np.random.permutation(n_train)
        val_size = int(0.1 * n_train)

        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        val_features_split = train_features[val_indices]
        val_labels_split = train_labels[val_indices]
        train_features = train_features[train_indices]
        train_labels = train_labels[train_indices]

    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(train_features),
        torch.LongTensor(train_labels)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(test_features),
        torch.LongTensor(test_labels)
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])

    # Step 3: Setup model and loss
    model = ImprovedCLIPClassifier().to(DEVICE)

    # Compute class weights
    class_counts = np.bincount(train_labels)
    total = len(train_labels)
    weights = torch.FloatTensor([total / (2 * c) for c in class_counts]).to(DEVICE)

    if use_focal_loss:
        criterion = FocalLossWithWeights(
            gamma=CONFIG['focal_gamma'],
            class_weights=weights,
            reduction='mean'
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=0.01
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['clip_epochs'],
        eta_min=1e-6
    )

    best_loss = float('inf')
    patience = 0
    best_state = None

    # Training loop
    for epoch in range(CONFIG['clip_epochs']):
        model.train()
        total_loss = 0

        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Early stopping based on training loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= CONFIG['early_stopping_patience']:
                break

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    # Step 4: Optimize threshold
    if use_threshold_opt:
        optimal_threshold, val_f1 = optimize_threshold(
            model, val_features_split, val_labels_split
        )
        print(f"    Optimal threshold: {optimal_threshold:.2f} (val F1: {val_f1:.3f})")
    else:
        optimal_threshold = 0.5

    # Step 5: Evaluate on test set
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = model(X)
            probs = F.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)

    # Apply optimized threshold
    y_pred = (y_probs >= optimal_threshold).astype(int)

    # Also compute metrics with default threshold for comparison
    y_pred_default = (y_probs >= 0.5).astype(int)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'optimal_threshold': float(optimal_threshold),
        # Default threshold metrics for comparison
        'accuracy_default': float(accuracy_score(y_true, y_pred_default)),
        'f1_default': float(f1_score(y_true, y_pred_default, zero_division=0)),
    }

    try:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_probs))
    except:
        metrics['roc_auc'] = 0.5

    # Cleanup
    del model, train_loader, test_loader, train_dataset, test_dataset
    gc.collect()

    return metrics


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def load_checkpoint(label_type):
    checkpoint_file = Path(CONFIG['checkpoint_path']) / f"checkpoint_{label_type}.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None


def save_checkpoint(results, label_type, completed):
    checkpoint = {
        'results': results,
        'completed': list(completed),
        'timestamp': datetime.now().isoformat()
    }
    checkpoint_file = Path(CONFIG['checkpoint_path']) / f"checkpoint_{label_type}.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    print("\n" + "="*70)
    print("IMPROVED 32-SUBJECT EXPERIMENT")
    print("SMOTE + Focal Loss + Threshold Optimization")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check if features exist
    feature_file = Path(CONFIG['features_path']) / "s01_features.npz"
    if not feature_file.exists():
        print("\nError: Pre-extracted features not found!")
        print(f"Please run run_ultra_efficient_experiment.py first to extract features.")
        return

    # Run LOSO for each task
    all_results = {}

    for label_type in ['valence', 'arousal']:
        print(f"\n{'='*70}")
        print(f"CLASSIFICATION TASK: {label_type.upper()}")
        print(f"{'='*70}")

        # Check checkpoint
        checkpoint = load_checkpoint(label_type)
        if checkpoint:
            results = checkpoint['results']
            completed = set(checkpoint['completed'])
            print(f"Resuming from checkpoint. Completed: {len(completed)}/32")
        else:
            results = {'per_subject': {}, 'config': CONFIG.copy()}
            completed = set()

        for test_subj in range(1, CONFIG['num_subjects'] + 1):
            if test_subj in completed:
                print(f"\nSubject {test_subj}/32: SKIPPED (already done)")
                continue

            print(f"\n--- Subject {test_subj}/32 ---")

            # Load test features
            test_feat, test_v, test_a = load_subject_features(test_subj)
            if test_feat is None:
                print(f"  Failed to load test subject")
                continue

            test_labels = test_v if label_type == 'valence' else test_a

            # Load training features
            train_feat_list = []
            train_labels_list = []

            for train_subj in range(1, CONFIG['num_subjects'] + 1):
                if train_subj == test_subj:
                    continue
                feat, v, a = load_subject_features(train_subj)
                if feat is not None:
                    train_feat_list.append(feat)
                    train_labels_list.append(v if label_type == 'valence' else a)

            train_features = np.concatenate(train_feat_list, axis=0)
            train_labels = np.concatenate(train_labels_list, axis=0)

            del train_feat_list, train_labels_list
            gc.collect()

            print(f"  Train: {len(train_labels)}, Test: {len(test_labels)}")
            print(f"  Test class dist: Pos={np.sum(test_labels)}, Neg={len(test_labels)-np.sum(test_labels)}")

            # Train with improvements
            print(f"  Training improved CLIP+MLP...")
            metrics = train_improved_model(
                train_features, train_labels,
                test_feat, test_labels,
                use_smote=CONFIG['use_smote'],
                use_focal_loss=CONFIG['use_focal_loss'],
                use_threshold_opt=CONFIG['use_threshold_optimization']
            )

            print(f"  Results: Acc={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f}")
            print(f"           (vs default threshold: Acc={metrics['accuracy_default']:.3f}, F1={metrics['f1_default']:.3f})")

            results['per_subject'][str(test_subj)] = metrics
            completed.add(test_subj)
            save_checkpoint(results, label_type, completed)

            del train_features, train_labels, test_feat, test_labels
            gc.collect()

        # Aggregate results
        accs = [results['per_subject'][str(s)]['accuracy'] for s in range(1, 33) if str(s) in results['per_subject']]
        f1s = [results['per_subject'][str(s)]['f1_score'] for s in range(1, 33) if str(s) in results['per_subject']]
        aucs = [results['per_subject'][str(s)].get('roc_auc', 0.5) for s in range(1, 33) if str(s) in results['per_subject']]
        thresholds = [results['per_subject'][str(s)].get('optimal_threshold', 0.5) for s in range(1, 33) if str(s) in results['per_subject']]

        # Default threshold comparison
        f1s_default = [results['per_subject'][str(s)].get('f1_default', 0) for s in range(1, 33) if str(s) in results['per_subject']]

        results['summary'] = {
            'mean_accuracy': float(np.mean(accs)),
            'std_accuracy': float(np.std(accs)),
            'mean_f1': float(np.mean(f1s)),
            'std_f1': float(np.std(f1s)),
            'mean_roc_auc': float(np.mean(aucs)),
            'std_roc_auc': float(np.std(aucs)),
            'mean_threshold': float(np.mean(thresholds)),
            'ci_95_acc': [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
            'ci_95_f1': [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
            # Comparison with default threshold
            'mean_f1_default': float(np.mean(f1s_default)),
            'f1_improvement': float(np.mean(f1s) - np.mean(f1s_default)),
        }

        # Baseline comparison (from previous experiment)
        baseline_f1 = 0.109 if label_type == 'valence' else 0.119
        baseline_acc = 0.746 if label_type == 'valence' else 0.691

        results['comparison'] = {
            'baseline_f1': baseline_f1,
            'baseline_acc': baseline_acc,
            'f1_improvement_vs_baseline': float(np.mean(f1s) - baseline_f1),
            'acc_change_vs_baseline': float(np.mean(accs) - baseline_acc),
            'relative_f1_improvement': float((np.mean(f1s) - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0,
        }

        print(f"\n{'='*50}")
        print(f"{label_type.upper()} FINAL RESULTS")
        print(f"{'='*50}")
        print(f"Accuracy: {results['summary']['mean_accuracy']:.3f} +/- {results['summary']['std_accuracy']:.3f}")
        print(f"F1 Score: {results['summary']['mean_f1']:.3f} +/- {results['summary']['std_f1']:.3f}")
        print(f"ROC-AUC:  {results['summary']['mean_roc_auc']:.3f} +/- {results['summary']['std_roc_auc']:.3f}")
        print(f"\nComparison with baseline (no improvements):")
        print(f"  F1: {baseline_f1:.3f} -> {results['summary']['mean_f1']:.3f} ({results['comparison']['relative_f1_improvement']:+.1f}%)")
        print(f"  Acc: {baseline_acc:.3f} -> {results['summary']['mean_accuracy']:.3f}")

        save_checkpoint(results, label_type, completed)
        all_results[label_type] = results

    # Save complete results
    complete_results = {
        'valence': all_results.get('valence', {}),
        'arousal': all_results.get('arousal', {}),
        'config': CONFIG,
        'timestamp': datetime.now().isoformat()
    }

    with open(f"{CONFIG['results_path']}/complete_results.json", 'w') as f:
        json.dump(complete_results, f, indent=2)

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print(f"Results saved to: {CONFIG['results_path']}/")
    print("="*70)

    # Summary comparison table
    print("\n" + "="*70)
    print("SUMMARY: BASELINE vs IMPROVED")
    print("="*70)
    print(f"{'Metric':<20} {'Baseline':<15} {'Improved':<15} {'Change':<15}")
    print("-"*65)

    if 'valence' in all_results:
        v_base_f1 = 0.109
        v_new_f1 = all_results['valence']['summary']['mean_f1']
        print(f"{'Valence F1':<20} {v_base_f1:<15.3f} {v_new_f1:<15.3f} {v_new_f1-v_base_f1:+.3f}")

    if 'arousal' in all_results:
        a_base_f1 = 0.119
        a_new_f1 = all_results['arousal']['summary']['mean_f1']
        print(f"{'Arousal F1':<20} {a_base_f1:<15.3f} {a_new_f1:<15.3f} {a_new_f1-a_base_f1:+.3f}")


if __name__ == "__main__":
    run_experiment()
