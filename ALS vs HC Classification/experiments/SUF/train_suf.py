import os
import re
import glob
import random
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

ALS_ROOT = "\IndicALS A Multi-lingual Indian ALS Speech Dataset\data\ALS"
HC_ROOT = "\IndicALS A Multi-lingual Indian ALS Speech Dataset\data\HC"
ALS_SPLIT_EXCEL = "\IndicALS A Multi-lingual Indian ALS Speech Dataset\data\ALS_5fold_split.xlsx"
HC_SPLIT_EXCEL = "\IndicALS A Multi-lingual Indian ALS Speech Dataset\data\HC_5fold_split.xlsx"

RESULTS_DIR = "results_SUF"
os.makedirs(RESULTS_DIR, exist_ok=True)

SR = 8000           # Sampling rate
FRAME_LEN = 160     # Micro-frame length (20ms at 8kHz)
MAX_LEN = 200       # Maximum sequential frames per 2s chunk

CHUNK_DURATION = 2.0
CHUNK_SAMPLES = int(CHUNK_DURATION * SR)  
HOP_SAMPLES = int(1.0 * SR)               

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
NUM_EPOCHS = 10


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def clean_subject_id(text):
    m = re.search(r'(\d+)', str(text))
    return m.group(1).zfill(5) if m else "00000"


def get_fold_subjects(df, fold_num, column):
    row = df[df['Fold'] == fold_num]
    if row.empty: 
        return []
    subj_str = str(row[column].values[0])
    return [clean_subject_id(s) for s in subj_str.split(',') if s.strip()]


def read_segments(txt_path):
    segments = []
    if not os.path.exists(txt_path): 
        return segments
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    segments.append((float(parts[0]), float(parts[1])))
                except ValueError: 
                    continue
    return segments


# ==========================================
# DATASET
# ==========================================
class SufRawDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        chunk_data = row["audio_chunk"]
        
        # Zero-mean, unit variance normalization
        chunk_data = (chunk_data - np.mean(chunk_data)) / (np.std(chunk_data) + 1e-8)
        
        # Frame signal into 160-sample steps
        framed_seg = librosa.util.frame(chunk_data, frame_length=FRAME_LEN, hop_length=80).T
        
        final = framed_seg[:MAX_LEN]
        if len(final) < MAX_LEN:
            final = np.pad(final, ((0, MAX_LEN - len(final)), (0, 0)))
            
        final = final[:, :, np.newaxis]
        return torch.tensor(final, dtype=torch.float32), torch.tensor(row["label"], dtype=torch.long)


# ==========================================
# MODEL ARCHITECTURE (Conv1D + LSTM)
# ==========================================
class ConvLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Spatial Feature Extractor
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=120)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.maxpool1 = nn.MaxPool1d(kernel_size=41)
        
        # Temporal Progression Feature Extractor
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=30, kernel_size=20, stride=1, padding=10)
        self.maxpool2 = nn.MaxPool1d(kernel_size=4)

        # Recurrent Network Core
        self.lstm1 = nn.LSTM(input_size=30, hidden_size=150, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=150, hidden_size=150, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=150, hidden_size=150, batch_first=True)

        # Classifier Output
        self.fc = nn.Linear(150, 2)

    def forward(self, x):
        batch_size, time_steps, sequence_length, channels = x.size()
        
        x = x.view(-1, sequence_length, channels)
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.batch_norm1(x)
        x = self.maxpool1(x)
        
        x = x.squeeze(-1)

        x = x.view(batch_size, time_steps, 256)
        x = x.permute(0, 2, 1)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        
        x = x.permute(0, 2, 1)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)

        x = self.fc(x[:, -1, :])
        return x


# ==========================================
# MAIN EXECUTABLE PIPELINE
# ==========================================
if __name__ == "__main__":
    
    all_processed_chunks = []
    target_phonemes = {"SSS", "SHS", "FFF"}

    # Data Loading and Preprocessing
    for root, label in [(ALS_ROOT, 1), (HC_ROOT, 0)]:
        sub_dirs = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
        for s_dir in tqdm(sub_dirs, desc=f"Processing SUF {'ALS' if label==1 else 'HC'}"):
            suf_path = os.path.join(s_dir, "SUF")
            if not os.path.isdir(suf_path):
                continue
                
            sid = clean_subject_id(os.path.basename(s_dir))
            
            for t_path in glob.glob(os.path.join(suf_path, "*.txt")):
                filename = os.path.basename(t_path)
                
                # Filter for specific sustained phoneme targets
                if not any(target in filename for target in target_phonemes):
                    continue  
                
                w_path = t_path.replace(".txt", ".wav")
                if not os.path.exists(w_path):
                    continue
                
                file_id = os.path.basename(w_path)
                    
                try:
                    audio, _ = librosa.load(w_path, sr=SR)
                    segments = read_segments(t_path)
                    
                    for start, end in segments:
                        start_sample = int(start * SR)
                        end_sample = int(end * SR)
                        word_audio = audio[start_sample:end_sample]
                        
                        if len(word_audio) == 0:
                            continue
                            
                        # Padding shorter segments or sliding window slicing longer segments
                        if len(word_audio) < CHUNK_SAMPLES:
                            padded_word = np.pad(word_audio, (0, CHUNK_SAMPLES - len(word_audio)), mode='constant')
                            all_processed_chunks.append({
                                "audio_chunk": padded_word,
                                "label": label,
                                "subject": sid,
                                "file_id": file_id
                            })
                        else:
                            word_frames = librosa.util.frame(
                                word_audio, 
                                frame_length=CHUNK_SAMPLES, 
                                hop_length=HOP_SAMPLES
                            ).T
                            for frame in word_frames:
                                all_processed_chunks.append({
                                    "audio_chunk": frame,
                                    "label": label,
                                    "subject": sid,
                                    "file_id": file_id
                                })
                except Exception:
                    continue

    master_df = pd.DataFrame(all_processed_chunks)

    if len(master_df) == 0:
        raise ValueError("No valid chunks extracted! Verify dataset directories and filename targets.")

    als_splits = pd.read_excel(ALS_SPLIT_EXCEL)
    HC_splits = pd.read_excel(HC_SPLIT_EXCEL)
    
    fold_accuracies, fold_f1_scores = [], []
    file_accuracies, file_f1_scores = [], []

    # 5-Fold Cross Validation
    for f in range(1, 6):
        tr_s = get_fold_subjects(als_splits, f, 'ALS_train') + get_fold_subjects(HC_splits, f, 'HC_train')
        vl_s = get_fold_subjects(als_splits, f, 'ALS_val') + get_fold_subjects(HC_splits, f, 'HC_val')
        te_s = get_fold_subjects(als_splits, f, 'ALS_test') + get_fold_subjects(HC_splits, f, 'HC_test')

        df_train = master_df[master_df['subject'].isin(tr_s)]
        df_val = master_df[master_df['subject'].isin(vl_s)]
        df_test = master_df[master_df['subject'].isin(te_s)]

        train_ds = SufRawDataset(df_train)
        val_ds = SufRawDataset(df_val)
        test_ds = SufRawDataset(df_test)

        tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        vl_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        te_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

        model = ConvLSTMModel().to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()

        best_f1 = 0
        epoch_log_file = open(f"{RESULTS_DIR}/fold_{f}_epoch_log.txt", "w")
        epoch_log_file.write("Epoch,Loss,Val_Acc,Val_F1\n")

        # Training Phase
        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0
            for x, y in tqdm(tr_loader, desc=f"Fold {f} Ep {epoch+1}", leave=False):
                opt.zero_grad()
                loss = crit(model(x.to(DEVICE)), y.to(DEVICE))
                loss.backward()
                opt.step()
                total_loss += loss.item()
            
            # Validation Step
            model.eval()
            y_v_true, y_v_pred = [], []
            with torch.no_grad():
                for x, y in vl_loader:
                    out = model(x.to(DEVICE))
                    y_v_true.extend(y.numpy())
                    y_v_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
            
            v_acc = accuracy_score(y_v_true, y_v_pred)
            v_f1 = f1_score(y_v_true, y_v_pred, zero_division=0)
            avg_loss = total_loss / len(tr_loader)
            
            epoch_log_file.write(f"{epoch+1},{avg_loss:.4f},{v_acc:.4f},{v_f1:.4f}\n")
            
            if v_f1 > best_f1:
                best_f1 = v_f1
                torch.save(model.state_dict(), f"{RESULTS_DIR}/best_model_fold_{f}.pt")

        epoch_log_file.close()

        # Test Evaluation
        if os.path.exists(f"{RESULTS_DIR}/best_model_fold_{f}.pt"):
            model.load_state_dict(torch.load(f"{RESULTS_DIR}/best_model_fold_{f}.pt"))
        model.eval()
        
        y_te_true, y_te_pred = [], []
        with torch.no_grad():
            for x, y in tqdm(te_loader, desc=f"Final Test Fold {f}"):
                out = model(x.to(DEVICE))
                y_te_true.extend(y.numpy())
                y_te_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
        
        # Chunk-Level Metrics
        t_acc = accuracy_score(y_te_true, y_te_pred)
        t_f1 = f1_score(y_te_true, y_te_pred, zero_division=0)
        t_cm = confusion_matrix(y_te_true, y_te_pred, labels=[0, 1])
        
        fold_accuracies.append(t_acc)
        fold_f1_scores.append(t_f1)

        # File-Level Evaluation via Majority Voting
        test_meta = df_test.reset_index(drop=True)
        test_meta['pred'] = y_te_pred
        
        file_true_labels = []
        file_pred_labels = []
        
        for file_name, group in test_meta.groupby('file_id'):
            true_label = group['label'].iloc[0]
            majority_prediction = group['pred'].value_counts().idxmax()
            
            file_true_labels.append(true_label)
            file_pred_labels.append(majority_prediction)
            
        f_acc = accuracy_score(file_true_labels, file_pred_labels)
        f_f1 = f1_score(file_true_labels, file_pred_labels, zero_division=0)
        f_cm = confusion_matrix(file_true_labels, file_pred_labels, labels=[0, 1])
        
        file_accuracies.append(f_acc)
        file_f1_scores.append(f_f1)

        # Save Fold Summary
        with open(f"{RESULTS_DIR}/fold_{f}_results.txt", "w") as rf:
            rf.write(f"Fold {f} Test Evaluation Summary\n")
            rf.write("="*35 + "\n")
            rf.write("[CHUNK-LEVEL PERFORMANCE]\n")
            rf.write(f"Accuracy: {t_acc:.4f}\n")
            rf.write(f"F1 Score: {t_f1:.4f}\n")
            rf.write(f"Confusion Matrix:\n{t_cm}\n\n")
            rf.write("[FILE-LEVEL MAJORITY VOTING PERFORMANCE]\n")
            rf.write(f"Accuracy: {f_acc:.4f}\n")
            rf.write(f"F1 Score: {f_f1:.4f}\n")
            rf.write(f"Confusion Matrix:\n{f_cm}\n")

    # Overall Summary Generation
    with open(f"{RESULTS_DIR}/final_summary.txt", "w") as sf:
        sf.write("Cross-Validation Complete Summary\n")
        sf.write("="*60 + "\n")
        sf.write(">> GLOBAL CHUNK-LEVEL STATS:\n")
        sf.write(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} (SD: {np.std(fold_accuracies):.4f})\n")
        sf.write(f"Mean F1-Score: {np.mean(fold_f1_scores):.4f} (SD: {np.std(fold_f1_scores):.4f})\n\n")
        
        sf.write(">> GLOBAL FILE-LEVEL STATS (Majority Voting):\n")
        sf.write(f"Mean Accuracy: {np.mean(file_accuracies):.4f} (SD: {np.std(file_accuracies):.4f})\n")
        sf.write(f"Mean F1-Score: {np.mean(file_f1_scores):.4f} (SD: {np.std(file_f1_scores):.4f})\n")
        sf.write("-" * 60 + "\n")
        
        for i in range(5):
            sf.write(f"Fold {i+1} -> Chunk Acc: {fold_accuracies[i]:.4f} | File Acc: {file_accuracies[i]:.4f}\n")

    print(f"\nTraining Complete. Global Cross-Validation Summary Saved to {RESULTS_DIR}")