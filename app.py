import streamlit as st
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
        
    def generate(self, idx, max_new_tokens, temperature=1.0):
        end_token_id = word_to_int['[END]']
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            if idx_next.item() == end_token_id:
                break
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx

# ==========================================
# 2. THE STREAMLIT USER INTERFACE
# ==========================================
# Configure the web page
st.set_page_config(page_title="BioBrick AI", page_icon="🧬")
st.title("🧬 AI Genetic Circuit Designer")
st.markdown("Design custom synthetic biology pathways using a miniature Large Language Model.")

# Load the model weights efficiently
@st.cache_resource 
def load_model():
    model = GeneticTransformer()
    model.load_state_dict(torch.load('biobrick_model.pth', weights_only=True))
    model.eval()
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'biobrick_model.pth' not found. Please ensure it is in the same folder as this script.")
    st.stop()

# Sidebar controls for the AI parameters
st.sidebar.header("Model Parameters")
temperature = st.sidebar.slider("Creativity (Temperature)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
st.sidebar.markdown("*Low = Strict & Logical. High = Mutated & Creative.*")

max_length = st.sidebar.slider("Max Circuit Length", min_value=3, max_value=15, value=10, step=1)

# Main interface inputs
# Main interface inputs
valid_promoters = ['pLac', 'pBAD', 'pTet', 'pLux', 'pTox', 'p35S', 'pCab', 'pCMV', 'pEF1a']
starting_part = st.selectbox("Choose a starting Promoter:", valid_promoters)

# The Generation Button
if st.button("Generate Genetic Circuit"):
    start_token = torch.tensor([[word_to_int[starting_part]]], dtype=torch.long)
    
    # Show a loading spinner while the AI "thinks"
    with st.spinner("AI is computing biological sequence..."):
        generated_ints = model.generate(start_token, max_new_tokens=max_length, temperature=temperature)[0].tolist()
        generated_circuit = [int_to_word[i] for i in generated_ints]
    
    st.success("Pathway Generated Successfully!")
    
    # Format and display the output beautifully
    formatted_circuit = " ➡️ ".join(generated_circuit)
    st.code(formatted_circuit, language="text")

   # ==========================================
    # 3. VISUAL DICTIONARY (Unified Breakdown)
    # ==========================================
    st.subheader("🧬 Circuit Breakdown")
    
    part_descriptions = {
        "pBAD": "Bacterial Promoter: Activated by the presence of arabinose sugar.",
        "pTet": "Bacterial Promoter: Repressed by the TetR protein.",
        "pLac": "Bacterial Promoter: Activated by lactose or IPTG.",
        "pLux": "Bacterial Promoter: Triggered by quorum sensing (population density).",
        "pTox": "Bacterial Promoter: Drives a toxin gene (cell death).",
        "p35S": "Plant Promoter: A strong, continuous promoter derived from a plant virus.",
        "pCab": "Plant Promoter: Activated specifically by light (essential for photosynthesis).",
        "pCMV": "Human Promoter: A highly active mammalian promoter.",
        "pEF1a": "Human Promoter: A stable, long-lasting mammalian expression promoter.",
        "RBS": "Bacterial/Plant Sequence: Ribosome Binding Site.",
        "Kozak": "Human Sequence: The mammalian equivalent of an RBS. Tells the ribosome where to start.",
        "GFP": "Coding Sequence: Green Fluorescent Protein (glows green under UV).",
        "RFP": "Coding Sequence: Red Fluorescent Protein (glows red under UV).",
        "Cas9": "Coding Sequence: CRISPR enzyme that cuts specific DNA sequences.",
        "lacZ": "Coding Sequence: Beta-galactosidase enzyme (breaks down lactose).",
        "lacY": "Coding Sequence: Lactose permease (brings lactose into the cell).",
        "lacA": "Coding Sequence: Galactoside acetyltransferase.",
        "luxA": "Coding Sequence: Luciferase alpha subunit (creates biological light).",
        "luxB": "Coding Sequence: Luciferase beta subunit (creates biological light).",
        "enzymeA": "Coding Sequence: Generic metabolic enzyme A.",
        "enzymeB": "Coding Sequence: Generic metabolic enzyme B.",
        "psbA": "Plant Gene: Codes for the core D1 protein of Photosystem II (captures light energy).",
        "psbD": "Plant Gene: Codes for the core D2 protein of Photosystem II.",
        "rbcL": "Plant Gene: Large subunit of RuBisCO (the enzyme that fixes carbon dioxide).",
        "rbcS": "Plant Gene: Small subunit of RuBisCO.",
        "INS": "Human Gene: Codes for the Insulin hormone, regulating blood sugar.",
        "p53": "Human Gene: The 'Guardian of the Genome', a crucial tumor suppressor protein.",
        "Term": "Terminator: Signals the machinery to stop transcribing."
    }

    # Loop through the AI's generated circuit and explain each part, avoiding duplicates
    explained_parts = set()
    for part in generated_circuit:
        if part in part_descriptions and part not in explained_parts:
            st.info(f"**{part}**: {part_descriptions[part]}")
            explained_parts.add(part)