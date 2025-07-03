import json
import os

# Paths
json_path = os.path.join('sktmt', 'vi.json')
out_dir = os.path.join('sktmt', 'vi')

# Ensure output directory exists
os.makedirs(out_dir, exist_ok=True)

# Read and process the large JSON file in a memory-efficient way
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    for key, value in data.items():
        # value is a list of strings; join them with newlines
        out_path = os.path.join(out_dir, f'{key}.txt')
        with open(out_path, 'w', encoding='utf-8') as out_f:
            out_f.write('\n'.join(value)) 