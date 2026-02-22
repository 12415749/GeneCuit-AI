import cobra
import cobra.io

print("🧫 Booting up E. coli Core Metabolic Digital Twin...")

# 1. Load the core textbook E. coli model provided by COBRApy
model = cobra.io.load_model("textbook")

# 2. Run Baseline FBA (Normal, un-engineered E. coli)
# The default objective is to maximize Biomass (growth)
baseline_solution = model.optimize()
print(f"\n🌱 BASELINE: Normal E. coli Growth Rate = {baseline_solution.objective_value:.4f} h^-1")

# The default energy required for the cell to just stay alive (ATP Maintenance) is 8.39 mmol/gDW/h
atpm_reaction = model.reactions.get_by_id("ATPM")
print(f"Normal ATP Maintenance requirement: {atpm_reaction.lower_bound}")

# ==========================================
# 3. SIMULATE THE AI'S GENETIC CIRCUIT
# ==========================================
print("\n🧬 Injecting AI-Generated Synthetic Circuit...")
print("Circuit: pLac -> RBS -> enzymeA -> RBS -> enzymeB -> Term")

# A synthetic circuit forces the cell to produce foreign proteins.
# This drains massive amounts of energy. We simulate this by artificially 
# cranking up the cell's minimum ATP Maintenance requirement.
synthetic_atp_cost = 45.0  # Simulating a heavy metabolic load
atpm_reaction.lower_bound = synthetic_atp_cost
print(f"New ATP requirement due to synthetic enzymes: {atpm_reaction.lower_bound}")

# 4. Run FBA again with the engineered constraints
engineered_solution = model.optimize()

# COBRApy will return a highly reduced growth rate, or 0.0 if the cell dies
if engineered_solution.objective_value < 0.01:
    print(f"\n⚠️ FATAL: Growth Rate = {engineered_solution.objective_value:.4f}")
    print("The AI's circuit is too metabolically expensive. The cell starved and died.")
else:
    print(f"\n✅ SUCCESS: Engineered Growth Rate = {engineered_solution.objective_value:.4f} h^-1")
    print("The E. coli successfully integrated the pathway, though it is growing slower.")