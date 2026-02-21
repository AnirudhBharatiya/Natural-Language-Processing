import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .q3_data import TextDataset

def train_nplm(model, train_data, val_data, vocab_size, unk_id,
               epochs=5, batch_size=64, learning_rate=0.001, save_path="nplm.pth"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_loader = DataLoader(TextDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data), batch_size=batch_size)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    history = {"train_loss": [], "val_loss": [], "train_ppl": [], "val_ppl": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for ctx, tgt in train_loader:
            ctx, tgt = ctx.to(device), tgt.to(device)
            
            # Masking with <UNK> 
            mask = torch.rand(ctx.shape, device=device) < 0.01
            ctx = ctx.masked_fill(mask, unk_id)
            
            optimizer.zero_grad()
            logits = model(ctx)
            loss = criterion(logits, tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train = total_loss / len(train_loader)
        train_ppl = math.exp(avg_train)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for ctx, tgt in val_loader:
                ctx, tgt = ctx.to(device), tgt.to(device)
                logits = model(ctx)
                val_loss += criterion(logits, tgt).item()
        
        avg_val = val_loss / len(val_loader)
        val_ppl = math.exp(avg_val)
        
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["train_ppl"].append(train_ppl)
        history["val_ppl"].append(val_ppl)
        
        print(f"Epoch {epoch+1} | Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f}")

    # Save Model
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'embedding_dim': model.embeddings.embedding_dim,
        'hidden_dim': model.hidden_layer[0].out_features,
        'context_size': model.input_dim // model.embeddings.embedding_dim,
    }, save_path)
    
    return history