import requests
import xml.etree.ElementTree as ET
import time

# A starter list of verified composite parts (devices/pathways) from iGEM
# You can easily expand this list to thousands of parts later!
composite_parts = [
    "BBa_I13507",  # Red Fluorescent Protein Reporter Device
    "BBa_I732006", # lacZ alpha reporter (Metabolism)
    "BBa_J245553", # GFP reporter
    "BBa_K173000", # T7 promoter -> GFP
    "BBa_K325909", # Lux reporter (Luminescence)
    "BBa_K3136004" # Multi-gene metabolic pathway
]

print("🧬 Starting iGEM Registry Scraper...")
dataset = []

for part_id in composite_parts:
    print(f"Fetching {part_id}...")
    # The official iGEM XML API URL
    url = f"http://parts.igem.org/xml/part.cgi?part={part_id}"
    
    try:
        response = requests.get(url)
        # Parse the raw XML data downloaded from the website
        root = ET.fromstring(response.content)
        
        # Navigate the XML tree to find the individual genetic parts in order
        circuit = []
        for subpart in root.findall(".//deep_subparts/subpart/part_name"):
            circuit.append(subpart.text)
        
        if circuit:
            # Format exactly how our LLM needs it: space-separated with an [END] token
            circuit_string = " ".join(circuit) + " [END]"
            dataset.append(circuit_string)
            print(f"  ✅ Extracted sequence: {circuit_string}")
        else:
            print(f"  ⚠️ No subparts found for {part_id}")
            
    except Exception as e:
        print(f"  ❌ Error fetching {part_id}: {e}")
        
    # Crucial: Pause for 1 second so we don't crash the iGEM servers and get blocked
    time.sleep(1) 

# Save the formatted sequences to a text file
with open("igem_dataset.txt", "w") as f:
    f.write("\n".join(dataset))

print("\n🎉 Scraping complete! The AI's new training data is saved to 'igem_dataset.txt'")