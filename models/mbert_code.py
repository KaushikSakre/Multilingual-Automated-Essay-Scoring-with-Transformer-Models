# 1) Import Libraries
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler, get_cosine_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import random

# # Set the seed for reproducibility
# def set_seed(seed_value=42):
#     random.seed(seed_value)    # Python random module
#     np.random.seed(seed_value) # Numpy module
#     torch.manual_seed(seed_value) # Torch
#     if torch.backends.cudnn.is_available():
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False

# set_seed()

# 2) Load Dataset
path = 'leearn.xlsx'
dataset = pd.read_excel(path)

# 3) Tokenizer and Data Splitting
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
train_df, temp_df = train_test_split(dataset, test_size=0.3, random_state=42, stratify=dataset['language'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['language'])
train_df = train_df.dropna(subset=['input_text'])
test_df = test_df.dropna(subset=['input_text'])
val_df = val_df.dropna(subset=['input_text'])

# 4) Class EssayDataset
class EssayDataset(Dataset):
    def __init__(self, texts, scores, languages, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        self.scores = torch.tensor(scores, dtype=torch.float)
        self.languages = languages
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.scores[idx]
        item['language'] = self.languages[idx]
        return item
    def __len__(self):
        return len(self.scores)

def calculate_qwk_rmse(val_preds, val_labels):
    val_preds_rounded = np.round(val_preds)
    val_labels_rounded = np.round(val_labels)
    overall_qwk = cohen_kappa_score(val_labels_rounded, val_preds_rounded, weights='quadratic')
    overall_rmse = np.sqrt(mean_squared_error(val_labels, val_preds))
    return overall_qwk, overall_rmse

# 5) Convert Data Into Lists and Create Dataset
train_dataset = EssayDataset(train_df['input_text'].tolist(), train_df['score'].tolist(), train_df['language'].tolist(), tokenizer)
val_dataset = EssayDataset(val_df['input_text'].tolist(), val_df['score'].tolist(), val_df['language'].tolist(), tokenizer)
test_dataset = EssayDataset(test_df['input_text'].tolist(), test_df['score'].tolist(), test_df['language'].tolist(), tokenizer)

# 6) Check the First Batch
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
for batch in train_loader:
    print(batch.keys())
    print(batch['language'])
    break

# Model Initialization
model = AutoModelForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=1)
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 10
epochs = list(range(1, num_epochs + 1))
num_training_steps = len(train_loader) * num_epochs
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Print CUDA usage
if device.type == 'cuda':
    print("Using CUDA: GPU acceleration is enabled.")
else:
    print("Using CPU: GPU acceleration is not enabled.")

# Early stopping and dynamic learning rate adjustment
early_stopping_patience = 3
best_val_loss = float('inf')
no_improve_epochs = 0

#Initialize Empty List
train_losses = []
val_losses = []
r2_scores = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'language'}
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}")

    # Validation Loss and QWK Calculation
    model.eval()
    total_val_loss = 0
    val_preds, val_labels, val_languages = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'language'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            val_loss = outputs.loss
            total_val_loss += val_loss.item()

            preds = outputs.logits.squeeze().cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())
            val_languages.extend(batch['language'])

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    val_preds_rounded = np.round(val_preds)
    val_labels_rounded = np.round(val_labels)
    overall_qwk_val = cohen_kappa_score(val_labels_rounded, val_preds_rounded, weights='quadratic')
    rmse_val = np.sqrt(mean_squared_error(val_labels_rounded, val_preds_rounded))
    r2_val = r2_score(val_labels_rounded, val_preds_rounded)
    r2_scores.append(r2_val)
    print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}, Validation QWK: {overall_qwk_val:.4f}, RMSE: {rmse_val:.4f}, R2: {r2_val:.4f}")


    # Check for early stopping and learning rate reduction
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= early_stopping_patience:
            print(f"Stopping early after {epoch + 1} epochs due to no improvement in validation loss.")
            break

    # Adjust learning rate if validation loss increases or plateaus
    if no_improve_epochs == 1:  # Start reducing LR after no improvement in one epoch
        new_lr = optimizer.param_groups[0]['lr'] * 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"Reduced learning rate to {new_lr:.8f} due to plateau in validation loss.")


# 11) Testing Phase and QWK Score Calculation
model.eval()
test_preds, test_labels, test_languages = [], [], []
with torch.no_grad():
    for batch in test_loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'language'}
        labels = batch['labels'].to(device)
        outputs = model(**inputs)
        preds = outputs.logits.squeeze().cpu().numpy()
        test_preds.extend(preds)
        test_labels.extend(labels.cpu().numpy())
        test_languages.extend(batch['language'])

test_preds_rounded = np.round(test_preds)
test_labels_rounded = np.round(test_labels)
overall_qwk_test = cohen_kappa_score(test_labels_rounded, test_preds_rounded, weights='quadratic')
rmse_test = np.sqrt(mean_squared_error(test_labels_rounded, test_preds_rounded))
r2_test = r2_score(test_labels_rounded, test_preds_rounded)
print(f"Testing Overall QWK Score: {overall_qwk_test:.4f}, RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}")

# Plotting the training loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label='Training Loss', marker='o')
plt.title('Training Loss Per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

#Plotting Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, val_losses, label='Validation Loss', marker='o', color='r')
plt.title('Validation Loss Per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

# Plotting R2 Scores
plt.figure(figsize=(10, 5))
plt.plot(epochs, r2_scores, label='R2 Score', marker='o', color='g')
plt.title('R2 Score Per Epoch')
plt.xlabel('Epochs')
plt.ylabel('R2 Score')
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()


# 12) Calculate and Plot QWK Scores
language_qwk_scores = {'Overall': overall_qwk_test}
unique_languages = set(test_languages)
for lang in unique_languages:
    indices = [i for i, l in enumerate(test_languages) if l == lang]
    lang_preds = [test_preds_rounded[i] for i in indices]
    lang_labels = [test_labels_rounded[i] for i in indices]
    lang_qwk = cohen_kappa_score(lang_labels, lang_preds, weights='quadratic')
    language_qwk_scores[lang.capitalize()] = lang_qwk


# Plotting QWK scores
plt.figure(figsize=(10, 6))
plt.bar(language_qwk_scores.keys(), language_qwk_scores.values(), color='skyblue')
plt.xlabel('Languages')
plt.ylabel('QWK Score')
plt.title('QWK Score Comparison Across Languages')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, (lang, score) in enumerate(language_qwk_scores.items()):
    plt.text(i, score + 0.02, f"{score:.2f}", ha='center', va='bottom', fontsize=12)
plt.savefig('qwk_score_comparison.png', dpi=300)
plt.show()

# 13) Calculate RMSE for Each Language
language_rmse_scores = {'Overall': rmse_test}
for lang in unique_languages:
    indices = [i for i, l in enumerate(test_languages) if l == lang]
    lang_preds = [test_preds_rounded[i] for i in indices]
    lang_labels = [test_labels_rounded[i] for i in indices]
    lang_rmse = np.sqrt(mean_squared_error(lang_labels, lang_preds))
    language_rmse_scores[lang.capitalize()] = lang_rmse

# Plotting RMSE scores
plt.figure(figsize=(10, 6))
plt.bar(language_rmse_scores.keys(), language_rmse_scores.values(), color='lightcoral')
plt.xlabel('Languages')
plt.ylabel('RMSE Score')
plt.title('RMSE Score Comparison Across Languages')
plt.ylim(0, max(language_rmse_scores.values()) + 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, (lang, score) in enumerate(language_rmse_scores.items()):
    plt.text(i, score + 0.02, f"{score:.2f}", ha='center', va='bottom', fontsize=12)
plt.savefig('rmse_score_comparison.png', dpi=300)
plt.show()

print("RMSE plot saved successfully!")

# Initialize the dictionary for R2 scores
language_r2_scores = {'Overall': r2_test}  # Assuming r2_test is already calculated for the overall test data

for lang in unique_languages:
    indices = [i for i, l in enumerate(test_languages) if l == lang]
    lang_preds = [test_preds_rounded[i] for i in indices]
    lang_labels = [test_labels_rounded[i] for i in indices]
    lang_r2 = r2_score(lang_labels, lang_preds)
    language_r2_scores[lang.capitalize()] = lang_r2


# Plotting R2 scores
plt.figure(figsize=(10, 6))
plt.bar(language_r2_scores.keys(), language_r2_scores.values(), color='lightgreen')
plt.xlabel('Languages')
plt.ylabel('R2 Score')
plt.title('R2 Score Comparison Across Languages')
plt.ylim(0, 1)  # Adjust this as needed based on your R2 score range
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, (lang, score) in enumerate(language_r2_scores.items()):
    plt.text(i, score + 0.02, f"{score:.2f}", ha='center', va='bottom', fontsize=12)
plt.savefig('r2_score_comparison.png', dpi=300)
plt.show()


# Save the trained model
model.save_pretrained('./mBERT')
tokenizer.save_pretrained('./mBERT')

print("Model saved successfully!")