import sys
import os

cols = ["viet", "chinese", "eval"]

INPUT_FILE = str(sys.argv[1])
OUTPUT_FILE = str(sys.argv[2])

def extract_data(file):
    # open xlsx file
    import pandas as pd
    data = pd.read_excel(file)
    viet = data[cols[0]].tolist()
    chinese = data[cols[1]].tolist()
    eval = data[cols[2]].tolist()
    res_data = []
    for i in range(len(eval)):
        if eval[i] != 'bad':
            res_data.append((viet[i], chinese[i]))
    # turn the data into a tsv file
    with open(OUTPUT_FILE, 'a', encoding='utf8') as f:
        for item in res_data:
            f.write("%s\t%s\n" % (item[1], item[0]))

if __name__ == "__main__":
    extract_data(INPUT_FILE)