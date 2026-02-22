import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import cobra
import cobra.io
import networkx as nx
import matplotlib.pyplot as plt

# ==========================================
# 1. RECREATE THE AI'S KNOWLEDGE BASE
# ==========================================
# Read the vocabulary dynamically from our dataset file
try:
    with open("igem_dataset.txt", "r") as f:
        text_data = f.read()
except FileNotFoundError:
    st.error("Dataset 'igem_dataset.txt' not found. Please run the data processor first.")
    st.stop()

words = text_data.split()
vocab = sorted(list(set(words)))
vocab_size = len(vocab)
word_to_int = {w: i for i, w in enumerate(vocab)}
int_to_word = {i: w for i, w in enumerate(vocab)}
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
st.markdown("Design custom synthetic biology pathways using a miniature Large Language Model by alienstar.com")

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
# Automatically find all promoters in the AI's vocabulary (words starting with 'p')
valid_promoters = [word for word in vocab if word.startswith('p')]
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


# ==========================================
    # 4. METABOLIC SIMULATION (FBA)
    # ==========================================
    st.markdown("---")
    st.subheader("🧫 Flux Balance Analysis (FBA) Simulation")
    
    with st.spinner("Booting up E. coli Digital Twin..."):
        try:
            # 1. Load the textbook E. coli model
            model = cobra.io.load_model("textbook")
            
            # 2. Run Baseline FBA
            baseline_solution = model.optimize()
            baseline_growth = baseline_solution.objective_value
            
            # 3. Calculate Dynamic Metabolic Load
            # Every part the AI generates costs the cell energy (ATP). 
            # We assign an arbitrary cost of 5.0 ATP per genetic part.
            atp_cost_per_part = 5.0
            synthetic_load = len(generated_circuit) * atp_cost_per_part
            
            # Apply the new heavy energy constraint to the cell
            atpm_reaction = model.reactions.get_by_id("ATPM")
            original_atpm = atpm_reaction.lower_bound
            atpm_reaction.lower_bound = original_atpm + synthetic_load
            
            # 4. Run Engineered FBA
            engineered_solution = model.optimize()
            engineered_growth = engineered_solution.objective_value
            
            # 5. Display the Results in a clean UI Dashboard
            cols = st.columns(3)
            cols[0].metric("Baseline Growth Rate", f"{baseline_growth:.4f} h^-1")
            cols[1].metric("Added ATP Burden", f"+{synthetic_load} units")
            
           # Check if the cell survived the AI's modification
            if engineered_growth < 0.01:
                cols[2].metric("Engineered Growth Rate", "FATAL", delta="-100%", delta_color="inverse")
                st.error("⚠️ **Simulation Failed:** The AI's circuit is too metabolically expensive. The cell starved and died.")
            else:
                # Calculate the percentage drop in growth speed
                delta_pct = ((engineered_growth - baseline_growth) / baseline_growth) * 100
                cols[2].metric("Engineered Growth Rate", f"{engineered_growth:.4f} h^-1", f"{delta_pct:.2f}%", delta_color="inverse")
                st.success("✅ **Simulation Success:** The E. coli successfully integrated the pathway and survived!")
                
                # ==========================================
                # 5. METABOLIC FLUX VISUALIZATION
                # ==========================================
                st.markdown("---")
                st.subheader("🕸️ Core Metabolic Flux Map")
                st.caption("Visualizing the top active chemical reactions in the engineered cell.")
                
                with st.spinner("Drawing metabolic network..."):
                    # Get the top 10 most active reactions (absolute flux)
                    top_fluxes = engineered_solution.fluxes.abs().sort_values(ascending=False).head(10)
                    
                    # Initialize a directed graph
                    G = nx.DiGraph()
                    
                    # Build the graph edges based on metabolites
                    for rxn_id in top_fluxes.index:
                        rxn = model.reactions.get_by_id(rxn_id)
                        # Connect each reactant to each product for this reaction
                        for reactant in rxn.reactants:
                            for product in rxn.products:
                                G.add_edge(reactant.id, product.id, weight=top_fluxes[rxn_id])
                    
                    # Set up the visual plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Create a layout that pushes nodes apart so they don't overlap
                    pos = nx.spring_layout(G, seed=42, k=5.0, iterations=100)
                    
                    # Draw the network
                    nx.draw(
                        G, pos, ax=ax, 
                        with_labels=True, 
                        node_color='#1f77b4', 
                        node_size=1500, 
                        font_size=8, 
                        font_color='white', 
                        font_weight='bold',
                        edge_color='#555555',
                        arrowsize=20
                    )
                    
                    # Make the background transparent to match Streamlit's dark mode
                    fig.patch.set_alpha(0.0)
                    ax.patch.set_alpha(0.0)
                    
                    # Display the plot in Streamlit
                    st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Simulation encountered an error: {e}")

# ==========================================
            # 5. METABOLIC FLUX VISUALIZATION
            # ==========================================
            '''if engineered_growth >= 0.01:
                st.markdown("---")
                st.subheader("🕸️ Core Metabolic Flux Map")
                st.caption("Visualizing the top active chemical reactions in the engineered cell.")
                
                with st.spinner("Drawing metabolic network..."):
                    # Get the top 10 most active reactions (absolute flux)
                    top_fluxes = engineered_solution.fluxes.abs().sort_values(ascending=False).head(10)
                    
                    # Initialize a directed graph
                    G = nx.DiGraph()
                    
                    # Build the graph edges based on metabolites
                    for rxn_id in top_fluxes.index:
                        rxn = model.reactions.get_by_id(rxn_id)
                        # Connect each reactant to each product for this reaction
                        for reactant in rxn.reactants:
                            for product in rxn.products:
                                # We use the flux value to determine the "weight" of the line
                                G.add_edge(reactant.id, product.id, weight=top_fluxes[rxn_id])
                    
                    # Set up the visual plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Create a layout that pushes nodes apart so they don't overlap
                    pos = nx.spring_layout(G, seed=42, k=5.0, iterations=100)
                    
                    # Draw the network
                    nx.draw(
                        G, pos, ax=ax, 
                        with_labels=True, 
                        node_color='#1f77b4', 
                        node_size=1500, 
                        font_size=8, 
                        font_color='white', 
                        font_weight='bold',
                        edge_color='#555555',
                        arrowsize=20
                    )
                    
                    # Make the background transparent to match Streamlit's dark mode
                    fig.patch.set_alpha(0.0)
                    ax.patch.set_alpha(0.0)
                    
                    # Display the plot in Streamlit
                    st.pyplot(fig)'''           