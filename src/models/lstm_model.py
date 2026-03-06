import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import mlflow
import joblib
import warnings
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.data_loader import load_splits
from src.models.experiment_tracker import setup_mlflow
from src.utils.config import LOOKBACK_WINDOW

warnings.filterwarnings('ignore')

class SequenceDataset(Dataset):
    """
    Converts tabular data into sequences for LSTM.
    Each sample = last LOOKBACK_WINDOW days of features
    Each label  = direction on day LOOKBACK_WINDOW+1 
    """
    def __init__(self, X: pd.DataFrame, y: pd.Series,
                 lookback: int = LOOKBACK_WINDOW):
        self.lookback = lookback
        self.X = torch.FloatTensor(X.values)
        self.y = torch.FloatTensor(y.values)

    def __len__(self):
        return len(self.X) - self.lookback

    def __getitem__(self, idx):
        # Sequence of past lookback days
        x_seq = self.X[idx : idx + self.lookback]
        # Label is the day AFTER the sequence
        label  = self.y[idx + self.lookback]
        return x_seq, label

class LSTMClassifier(nn.Module):
    """
    2-layer LSTM with dropout for stock direction prediction.
    Input:  (batch, lookback, n_features)
    Output: (batch, 1) — probability of up move
    """
    def __init__(self, input_size, hidden_size=64,
                 num_layers=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) 
        
        # Fully connected output
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch, lookback, features)
        lstm_out, _ = self.lstm(x)
        
        # Take only the last timestep output
        last_out = lstm_out[:, -1, :]
        out = self.dropout(last_out)
        
        return self.fc(out).squeeze(1)


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        
        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    """Evaluate model. Returns AUC and predictions."""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch  = X_batch.to(device)
            probs    = model(X_batch).cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())
            
    auc = roc_auc_score(all_labels, all_probs)
    preds = (np.array(all_probs) > 0.5).astype(int)
    
    return auc, np.array(all_probs), np.array(all_labels), preds


def train_lstm(ticker='AAPL', epochs=50, batch_size=32,
               hidden_size=64, num_layers=2, dropout=0.3,
               learning_rate=0.001, lookback=LOOKBACK_WINDOW):
    """Full LSTM training pipeline."""
    setup_mlflow()
    print(f'Training LSTM for {ticker}...')
    print(f'Lookback window: {lookback} days')
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(ticker)
    n_features = X_train.shape[1]
    print(f'Features: {n_features}')
    
    # Create datasets
    train_ds = SequenceDataset(X_train, y_train, lookback)
    val_ds   = SequenceDataset(X_val,   y_val,   lookback)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    
    # Device — CPU for your setup
    device = torch.device('cpu')
    print(f'Device: {device}') 

    # Model
    model = LSTMClassifier(
        input_size=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Class-weighted loss
    pos_weight = torch.tensor([float((y_train==0).sum()/(y_train==1).sum())])
    criterion  = nn.BCELoss()
    optimizer  = optim.Adam(model.parameters(), lr=learning_rate,
                            weight_decay=1e-5)
    
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=False)
    
    # Training loop with early stopping
    best_val_auc  = 0
    patience_count = 0
    patience_limit = 10
    best_model_state = None
    history = {'train_loss': [], 'val_auc': []}
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader,
                                 optimizer, criterion, device)
        val_auc, _, _, _ = evaluate(model, val_loader, device)
        
        scheduler.step(val_auc)
        history['train_loss'].append(train_loss)
        history['val_auc'].append(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc     = val_auc
            best_model_state = model.state_dict().copy()
            patience_count   = 0
        else:
            patience_count += 1
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:3d} | Loss: {train_loss:.4f} '
                  f'| Val AUC: {val_auc:.4f} | Best: {best_val_auc:.4f}')
            
        if patience_count >= patience_limit:
            print(f'Early stopping at epoch {epoch+1}')
            break 

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    print(f'Best Val AUC: {best_val_auc:.4f}')
    
    # Log to MLflow
    params = {
        'ticker': ticker, 'epochs': epochs,
        'hidden_size': hidden_size, 'num_layers': num_layers,
        'dropout': dropout, 'learning_rate': learning_rate,
        'lookback': lookback, 'batch_size': batch_size
    }
    
    with mlflow.start_run(run_name=f'LSTM_{ticker}'):
        mlflow.log_params(params)
        mlflow.log_metrics({'val_auc': best_val_auc})
        mlflow.pytorch.log_model(model, 'lstm_model')
        
    # Save model
    save_dir = Path('experiments/models')
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / f'lstm_{ticker}.pt')
    torch.save(model, save_dir / f'lstm_{ticker}_full.pt')
    print(f'LSTM saved!')
    
    return model, history, best_val_auc

if __name__ == '__main__':
    model, history, best_auc = train_lstm('AAPL', epochs=50)
    print(f'Final best AUC: {best_auc:.4f}')