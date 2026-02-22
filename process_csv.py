import pandas as pd

print("📊 Loading BioBrick CSV database...")

# 1. Read the CSV file using pandas
try:
    df = pd.read_csv('biobricks.csv')
except FileNotFoundError:
    print("❌ Error: biobricks.csv not found in this folder.")
    exit()

dataset = []

# 2. Loop through the 'Sequence' column
for sequence in df['Sequence']:
    # Format exactly how our LLM needs it
    formatted_seq = f"{sequence} [END]"
    dataset.append(formatted_seq)
    print(f"  ✅ Added: {formatted_seq}")

# 3. Save the formatted sequences to our training text file
with open("igem_dataset.txt", "w") as f:
    f.write("\n".join(dataset))

print("\n🎉 CSV processing complete! Run 'python train_llm.py' to train the AI.")