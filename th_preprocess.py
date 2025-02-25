import re

processed = []

with open(f'th/chinese_pars.txt', 'r', encoding='utf-8') as f:
    first_num_re = r"^(\d+ \.)"
    # Remove the first number in each line
    for line in f:
        line = re.sub(first_num_re, '', line)
        processed.append(line)

with open(f'th/chinese_pars_processed.txt', 'w', encoding='utf-8') as f:
    for line in processed:
        f.write(line)

trans_processed = []

with open(f'th/translation_pars.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()
    # Remove empty lines
    data = [line for line in data if line.strip()]
    for line in data:
        trans_processed.append(line)

with open(f'th/translation_pars_processed.txt', 'w', encoding='utf-8') as f:
    for line in trans_processed:
        f.write(line)