# Open "aligned_dvsktt.xlsx", extract the 'cn_sv_dict'
# and save it as 'cn_sv_dict.json'

import pandas as pd
import json

# Load the data
df = pd.read_excel('aligned_dvsktt.xlsx')
data_col = df['cn_sv_dict']

def string_to_set(text_string):
    text_string = text_string.strip("{}")
    if not text_string:
        return {}
    pairs = text_string.split(", ")
    result = {}
    for pair in pairs:
        key, value = pair.split(": ")
        result[key.strip("'")] = value.strip("'")
    return result

# Extract the 'cn_sv_dict'
cn_sv_dict = {}
for data in data_col:
    data = string_to_set(data)
    cn_sv_dict.update((key.replace(' ', ''), value) for key, value in data.items())

# Save the 'cn_sv_dict' as 'cn_sv_dict.json'
with open('cn_sv_dict.json', 'w', encoding='utf-8') as f:
    json.dump(cn_sv_dict, f, ensure_ascii=False, indent=4)