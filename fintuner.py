import os
import gc
import random
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from config import *
from data_loader import load_data
from model.bert_projector import BertProjector
from model.gnn import GNN
from utils.graph_builder import build_base_graph
from utils.knn_augment import add_knn_edges_to_graph
from torch_geometric.utils import from_networkx, to_undirected

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# === LOAD DATA ===
df, le = load_data(MAX_SAMPLES)
num_classes = len(le.classes_)
logger.info(f"Num classes after filtering: {num_classes}")
gc.collect()

# === BUILD GRAPH ===
G_base = build_base_graph(df)
node_list = list(G_base.nodes())
_df_indexed = df.set_index('id')
_df_sel = _df_indexed.loc[[n for n in node_list if n in _df_indexed.index]]
df_reindexed = _df_sel.reset_index()

graph_data = from_networkx(G_base)
graph_data.y = torch.tensor(df_reindexed['label_enc'].values, dtype=torch.long)

texts = df_reindexed['text'].tolist()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# === BERT MODEL ===
lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["query", "key", "value"], lora_dropout=0.05, bias="none")
bert_model = BertProjector(MODEL_NAME, num_classes, GNN_DIM).to(DEVICE)

try:
    bert_model.bert = get_peft_model(bert_model.bert, lora_cfg)
except Exception as e:
    logger.warning(f"PEFT failed, trying fallback targets: {e}")
    lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_lin", "k_lin", "v_lin"], lora_dropout=0.05, bias="none")
    bert_model.bert = get_peft_model(bert_model.bert, lora_cfg)

try:
    bert_model.bert.print_trainable_parameters()
except Exception:
    logger.info("(PEFT) print_trainable_parameters not available")

# === LOSSES ===
def contrastive_loss(emb, labels, margin=0.5):
    sim = F.cosine_similarity(emb.unsqueeze(1), emb.unsqueeze(0), dim=-1)
    label_mat = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    pos = label_mat * (1.0 - sim)
    neg = (1.0 - label_mat) * torch.relu(sim - margin)
    denom = (label_mat.sum() + (1.0 - label_mat).sum()).clamp(min=1.0)
    return (pos.sum() + neg.sum()) / denom

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    def forward(self, logits, target):
        confidence = 1.0 - self.smoothing
        log_probs = F.log_softmax(logits, dim=-1)
        n_classes = logits.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / max(1, (n_classes - 1)))
            true_dist.scatter_(1, target.unsqueeze(1), confidence)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

# === DATASET ===
class TextDataset(Dataset):
    def __init__(self, texts): self.texts = texts
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return self.texts[i]

def get_scheduler_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)

# === FINE-TUNE BERT ===
def fine_tune_bert(model, texts, labels, batch_size=BERT_BATCH_SIZE, cont_weight=CONT_WEIGHT, accum_steps=ACCUM_STEPS):
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=BERT_LR, weight_decay=1e-2, eps=1e-8)
    indices = np.arange(len(texts))
    y_np = labels.cpu().numpy()
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=y_np) if len(np.unique(y_np))>1 else (indices, [])
    total_steps = max(1, (len(train_idx) // batch_size // accum_steps) * EPOCHS_BERT)
    warmup_steps = max(1, int(0.10 * total_steps))
    scheduler = get_scheduler_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss()
    scaler = amp.GradScaler()

    losses, accs = [], []
    step = 0
    for epoch in range(EPOCHS_BERT):
        model.train()
        np.random.shuffle(train_idx)
        epoch_loss = 0.0
        batch_count = 0
        prog = tqdm(range(0, len(train_idx), batch_size), desc=f"BERT Epoch {epoch+1}/{EPOCHS_BERT}")
        for i in prog:
            batch_idx = train_idx[i:i+batch_size]
            batch_txt = [texts[j] for j in batch_idx]
            batch_y = torch.tensor(y_np[batch_idx], dtype=torch.long, device=DEVICE)

            enc = tokenizer(batch_txt, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors='pt')
            enc = {k: v.to(DEVICE) for k, v in enc.items()}

            with amp.autocast():
                logits, proj = model(enc['input_ids'], enc['attention_mask'], project=True)
                cls_loss = criterion(logits, batch_y)
                cont_loss = contrastive_loss(proj, batch_y) if cont_weight > 0 else torch.tensor(0.0, device=DEVICE)
                loss = cls_loss + cont_weight * cont_loss
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if ((batch_count + 1) % accum_steps) == 0 or (i + batch_size) >= len(train_idx):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

            epoch_loss += loss.item() * accum_steps
            batch_count += 1
            prog.set_postfix(loss=loss.item() * accum_steps)
            del enc, logits, proj, batch_y
            gc.collect(); torch.cuda.empty_cache()

        avg_loss = epoch_loss / max(1, batch_count)
        losses.append(avg_loss)

        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad(), amp.autocast():
            for i in range(0, len(val_idx), batch_size):
                idx = val_idx[i:i+batch_size]
                txt = [texts[j] for j in idx]
                y = y_np[idx]
                enc = tokenizer(txt, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors='pt')
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                logits, _ = model(enc['input_ids'], enc['attention_mask'], project=False)
                pred = logits.argmax(dim=1).cpu().numpy()
                all_pred.extend(pred); all_true.extend(y)
                del enc, logits
                gc.collect(); torch.cuda.empty_cache()

        if len(all_true) == 0:
            acc = f1 = 0.0
        else:
            acc = accuracy_score(all_true, all_pred)
            f1 = f1_score(all_true, all_pred, average='macro')
        accs.append(acc)
        logger.info(f"[BERT] Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f} | F1: {f1:.4f}")

        ckpt_path = os.path.join(MODEL_SAVE_PATH, f"bert_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)

    return losses, accs

labels_tensor = torch.tensor(df_reindexed['label_enc'].values, dtype=torch.long, device=DEVICE)
bert_losses, bert_accs = fine_tune_bert(bert_model, texts, labels_tensor, cont_weight=CONT_WEIGHT)

# === PROJECT EMBEDDINGS ===
def get_projected_embeddings(model, texts, batch_size=BERT_BATCH_SIZE):
    model.eval()
    ds = TextDataset(texts)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=True)
    embs = []
    with torch.no_grad(), amp.autocast():
        for batch in tqdm(dl, desc="Projecting SciBERT"):
            enc = tokenizer(batch, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors='pt')
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            _, proj = model(enc['input_ids'], enc['attention_mask'], project=True)
            embs.append(proj.cpu().to(torch.float32))
            del enc, proj
            gc.collect(); torch.cuda.empty_cache()
    return torch.cat(embs, dim=0) if embs else torch.zeros((0, GNN_DIM))

proj_emb_cpu = get_projected_embeddings(bert_model, texts)
proj_emb_cpu = proj_emb_cpu / (proj_emb_cpu.norm(dim=1, keepdim=True) + 1e-12)
proj_emb_np = proj_emb_cpu.numpy()

# === KNN GRAPH ===
G_aug = add_knn_edges_to_graph(G_base, node_list, proj_emb_cpu, k=KNN_K)
pyg_graph = from_networkx(G_aug)
pyg_graph.y = torch.tensor(df_reindexed['label_enc'].values, dtype=torch.long, device=DEVICE)
edge_index = to_undirected(pyg_graph.edge_index).to(DEVICE)
node_emb = proj_emb_cpu.to(DEVICE)

# === GNN TRAINING ===
gnn = GNN(GNN_DIM, 128, num_classes, dropout=0.3).to(DEVICE)
opt_gnn = AdamW(gnn.parameters(), lr=GNN_LR, weight_decay=1e-2)
criterion_gnn = LabelSmoothingCrossEntropy(smoothing=0.1)

idx = np.arange(len(df_reindexed))
y_np = pyg_graph.y.cpu().numpy()
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y_np)
train_mask = torch.zeros(len(df_reindexed), dtype=torch.bool, device=DEVICE)
test_mask  = torch.zeros(len(df_reindexed), dtype=torch.bool, device=DEVICE)
train_mask[torch.tensor(train_idx, dtype=torch.long, device=DEVICE)] = True
test_mask[torch.tensor(test_idx, dtype=torch.long, device=DEVICE)] = True

scaler_gnn = amp.GradScaler()
gnn_losses, gnn_accs = [], []

for epoch in range(EPOCHS_GNN):
    gnn.train()
    opt_gnn.zero_grad()
    with amp.autocast():
        logits = gnn(node_emb, edge_index)
        loss = criterion_gnn(logits[train_mask], pyg_graph.y[train_mask])
    scaler_gnn.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(gnn.parameters(), max_norm=1.0)
    scaler_gnn.step(opt_gnn)
    scaler_gnn.update()
    gnn_losses.append(loss.item())

    gnn.eval()
    with torch.no_grad():
        logits = gnn(node_emb, edge_index)
        pred = logits[test_mask].argmax(dim=1).cpu().numpy()
        true = pyg_graph.y[test_mask].cpu().numpy()
        acc = accuracy_score(true, pred)
        gnn_accs.append(acc)

    if epoch % 5 == 0 or epoch == EPOCHS_GNN-1:
        logger.info(f"[GNN] Epoch {epoch+1}/{EPOCHS_GNN} | Loss: {loss.item():.4f} | Test Acc: {acc:.4f}")

    if (epoch + 1) % 5 == 0:
        ckpt = os.path.join(MODEL_SAVE_PATH, f"gnn_epoch{epoch+1}.pt")
        torch.save(gnn.state_dict(), ckpt)

torch.save(gnn.state_dict(), os.path.join(MODEL_SAVE_PATH, "gnn_final.pt"))

# === FINAL EMBEDDINGS ===
gnn.eval()
with torch.no_grad():
    refined_emb = gnn.embed(node_emb, edge_index)
refined_emb_np = refined_emb.cpu().numpy().astype(np.float32)
refined_emb_np = refined_emb_np / (np.linalg.norm(refined_emb_np, axis=1, keepdims=True) + 1e-12)

# === RECOMMEND FUNCTION ===
def recommend(query, top_k=5, use_refined=True):
    enc = tokenizer([query], truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors='pt')
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        _, q_proj = bert_model(enc['input_ids'], enc['attention_mask'], project=True)
        q = q_proj.cpu().numpy().astype(np.float32)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    emb_matrix = refined_emb_np if use_refined else proj_emb_np
    sims = cosine_similarity(q, emb_matrix)[0]
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return [node_list[i] for i in top_idx], sims[top_idx]

# === PRETTY PRINT ===
def pretty_recommendation(query, ids, scores, df):
    df_lookup = df.set_index('id')
    lines = []
    for pid, sc in zip(ids, scores):
        row = df_lookup.loc[pid]
        lines.append(f"  • {row['title']}\n    Abstract: {row['abstract'][:180].rstrip()}...\n    Score: {sc:.4f}\n")
    return f"\nRecommendations for \"{query}\":\n" + "\n".join(lines)

# === DEMO ===
if __name__ == "__main__":
    query = "advancements in graph neural networks for computer vision"
    rec_ids, rec_scores = recommend(query, top_k=5)

    print("\n" + "="*70)
    print(" " * 20 + "TRAINING SUMMARY")
    print("="*70)
    print("\nSciBERT Fine-tuning (LoRA r=8)")
    for ep, (l, a) in enumerate(zip(bert_losses, bert_accs), 1):
        print(f"   Epoch {ep:2d} | Loss: {l:.4f} | Val Acc: {a:.4f}")

    print("\nGNN (GraphSAGE) on SciBERT + KNN graph")
    for ep in range(0, len(gnn_losses), 5):
        epoch = ep + 1
        print(f"   Epoch {epoch:2d} | Loss: {gnn_losses[ep]:.4f} | Test Acc: {gnn_accs[ep]:.4f}")
    print(f"   Final     | Loss: {gnn_losses[-1]:.4f} | Test Acc: {gnn_accs[-1]:.4f}")
    print("="*70)
    print(pretty_recommendation(query, rec_ids, rec_scores, df_reindexed))
    print("="*70)