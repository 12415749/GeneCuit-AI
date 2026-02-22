import torch
import torch.nn as nn
from torch.nn import functional as F

# ==========================================
# 1. THE UNIFIED DATASET (Bacteria + Plant + Human)
# ==========================================
text_data = """
pBAD RBS GFP Term [END]
pTet RBS RFP Term [END]
pLac RBS Cas9 Term [END]
pLac RBS lacZ RBS lacY RBS lacA Term [END]
pLux RBS luxA RBS luxB Term [END]
pTox RBS enzymeA RBS enzymeB Term [END]
pBAD RBS luxA RBS luxB Term [END]
p35S RBS psbA RBS psbD Term [END]
pCab RBS rbcL RBS rbcS Term [END]
pCMV Kozak INS Term [END]
pEF1a Kozak p53 Term [END]
p35S RBS rbcL Term [END]
pCMV Kozak p53 Term [END]
"""

words = text_data.split()
vocab = sorted(list(set(words)))
vocab_size = len(vocab)
word_to_int = {w: i for i, w in enumerate(vocab)}
int_to_word = {i: w for i, w in enumerate(vocab)}
data = torch.tensor([word_to_int[w] for w in words], dtype=torch.long)

# NEW: Transformer Hyperparameters
block_size = 8    # Context window: how many BioBricks the AI looks at simultaneously
embed_size = 16   # The size of our dense mathematical vectors
batch_size = 8    # How many circuits we process at once during training

def get_batch():
    # Grab random chunks of the dataset to train on
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# ==========================================
# 2. THE SELF-ATTENTION LAYER
# ==========================================
class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        # A mask to hide future genetic parts from the AI so it doesn't "cheat"
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        
        # Calculate Attention Scores (Q * K^T / sqrt(d_k))
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) 
        
        # Apply the mask and calculate Softmax
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) 
        
        # Multiply weights by Values
        v = self.value(x)
        out = wei @ v
        return out

# ==========================================
# 3. THE TRANSFORMER MODEL
# ==========================================
class GeneticTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(block_size, embed_size) # NEW: Positional awareness
        self.sa_head = Head(embed_size) # NEW: Injecting our Attention Head
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) # What is the biological part?
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # Where is it located?
        
        x = tok_emb + pos_emb # Combine identity and position
        x = self.sa_head(x)   # Pass through Self-Attention
        logits = self.lm_head(x) # Predict next token
        
        return logits
        
    def generate(self, idx, max_new_tokens):
        # Look up the mathematical ID for our new [END] token
        end_token_id = word_to_int['[END]']
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # THE KILL SWITCH: If the AI generates [END], stop the loop immediately!
            if idx_next.item() == end_token_id:
                break
                
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx

# ==========================================
# 4. TRAINING & INFERENCE
# ==========================================
model = GeneticTransformer()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

print("Training the Miniature Transformer...")
for steps in range(1000):
    xb, yb = get_batch()
    
    logits = model(xb)
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    targets = yb.view(B*T)
    
    loss = F.cross_entropy(logits, targets)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"Training Complete! Final Loss: {loss.item():.4f}\n")

# Start sequence: 'pLux' (promoter for luminescence)
start_token = torch.tensor([[word_to_int['pLux']]], dtype=torch.long)

# Increase max_new_tokens to 5 to give it room to build a multi-gene pathway
generated_ints = model.generate(start_token, max_new_tokens=5)[0].tolist()

generated_circuit = [int_to_word[i] for i in generated_ints]
print("🤖 Transformer-Generated Metabolic Pathway:")
print(" -> ".join(generated_circuit))


# Prompt it with 'pLac'
start_token = torch.tensor([[word_to_int['pLac']]], dtype=torch.long)

# We give it a max of 10, but the [END] token should stop it early!
generated_ints = model.generate(start_token, max_new_tokens=10)[0].tolist()
generated_circuit = [int_to_word[i] for i in generated_ints]

print("🤖 AI-Generated Circuit with Stop Logic:")
print(" -> ".join(generated_circuit))


# ==========================================
# 6. SAVING THE MODEL WEIGHTS
# ==========================================
# Save the model's learned parameters to a file
torch.save(model.state_dict(), 'biobrick_model.pth')
print("💾 Model saved successfully as 'biobrick_model.pth'")


