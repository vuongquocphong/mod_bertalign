import bertalign
from bertalign import Encoder

src_par = ""
tgt_par = ""

with open("./data/Complete/chinese_val_snts.txt", "r", encoding="utf8") as f:
    lines = f.readlines()
    for line in lines:
        src_par += line.strip() + "\n"

with open("./data/Complete/translation_val_snts.txt", "r", encoding="utf8") as f:
    lines = f.readlines()
    for line in lines:
        tgt_par += line.strip() + "\n"

aligner = bertalign.Bertalign(src=src_par, tgt=tgt_par, is_split=True)

aligner.align_sents()

alignments = []
for bead in (aligner.result):
    src_line = aligner._get_line(bead[0], aligner.src_sents)
    tgt_line = aligner._get_line(bead[1], aligner.tgt_sents)
    # calculate similarity
    alignments.append((src_line, tgt_line))

golden = []
with open("./data/Complete/real_golden.txt", "r", encoding="utf-8") as f:
    data = f.readlines()
    for i in range(len(data)):
        data[i] = data[i].strip()
        first_part = ""
        second_part = ""
        for char in data[i]:
            if char == "\t":
                second_part = data[i][data[i].index(char) + 1:]
                break
            first_part += char
        golden.append((first_part, second_part))

match = 0
for alignment in alignments:
    for gold in golden:
        if alignment[0] == gold[0] and alignment[1] == gold[1]:
            match += 1
            break

precision = match / len(alignments) if len(alignments) > 0 else 0
recall = match / len(golden) if len(golden) > 0 else 0

with open("./data/Complete/result.txt", "a", encoding="utf-8") as f:
    f.write(f"Align with trans source snts using bertalign baseline" + "\n")
    f.write("Precision: " + str(precision) + "\n")
    f.write("Recall: " + str(recall) + "\n")
    f.write("F1: " + str(2 * precision * recall / (precision + recall)) + "\n")
    f.write("------------------\n")