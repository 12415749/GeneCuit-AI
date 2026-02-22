import torch
import torch.nn as nn
from torch.nn import functional as F

# 1. Recreate the Vocabulary (MUST match the training data exactly!)
text_data = """
pBAD RBS GFP Term [END]
pTet RBS RFP Term [END]
pLac RBS Cas9 Term [END]
pLac RBS lacZ RBS lacY RBS lacA Term [END]
pLux RBS luxA RBS luxB Term [END]
pTox RBS enzymeA RBS enzymeB Term [END]
pBAD RBS luxA RBS luxB Term [END]
"""
words = text_data.split()
vocab = sorted(list(set(words)))
vocab_size = len(vocab)
word_to_int = {w: i for i, w in enumerate(vocab)}
int_to_word = {i: w for i, w in enumerate(vocab)}

# 2. Recreate the Transformer Architecture (Must match exactly!)
block_size = 8
embed_size = 16

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) 
        v = self.value(x)
        return wei @ v

class GeneticTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(block_size, embed_size) 
        self.sa_head = Head(embed_size) 
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) 
        x = tok_emb + pos_emb 
        x = self.sa_head(x)   
        logits = self.lm_head(x) 
        return logits
        
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx

# ==========================================
# 3. LOAD THE BRAIN AND GENERATE
# ==========================================
# Initialize the empty model
loaded_model = GeneticTransformer()

# Load the saved memory (.pth file)
loaded_model.load_state_dict(torch.load('biobrick_model.pth'))
loaded_model.eval() # Set the model to evaluation (inference) mode

print("🧠 Model loaded successfully! Let's build a circuit.")

# Prompt it with 'pLac'
start_token = torch.tensor([[word_to_int['pLac']]], dtype=torch.long)
generated_ints = loaded_model.generate(start_token, max_new_tokens=6)[0].tolist()
generated_circuit = [int_to_word[i] for i in generated_ints]

print("🤖 AI-Generated Circuit from Loaded Memory:")
print(" -> ".join(generated_circuit))