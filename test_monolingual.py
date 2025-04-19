import sys
from bertalign_trans import Encoder
import bertalign_trans

src_snts = []
src_trans_snts = []

with open("./data/dai_viet_su_ki/chinese_snts.txt", "r", encoding="utf-8") as f:
    for line in f:
        src_snts.append(line.strip())

with open("./data/dai_viet_su_ki/trans_src_snts.txt", "r", encoding="utf-8") as f:
    for line in f:
        src_trans_snts.append(line.strip())

map_src_trans = {}
for i in range(len(src_snts)):
    map_src_trans[src_trans_snts[i]] = src_snts[i]

model = Encoder("LaBSE")

src_pars = ""
for src_snt in src_snts:
    src_pars += src_snt + "\n"

src_trans_pars = ""
for src_trans_snt in src_trans_snts:
    src_trans_pars += src_trans_snt + "\n"

trans_pars = ""
with open("./data/dai_viet_su_ki/translation_snts.txt", "r", encoding="utf-8") as f:
    for line in f:
        trans_pars += line.strip() + "\n"

alignments = []
aligner = bertalign_trans.Bertalign(src=src_pars, translated_src=src_trans_pars, tgt=trans_pars, model=model, is_split=True)
aligner.align_sents()
for bead in (aligner.result):
    src_line = aligner._get_line(bead[0], aligner.src_sents)
    tgt_line = aligner._get_line(bead[1], aligner.tgt_sents)
    # calculate similarity
    alignments.append((src_line, tgt_line))

golden = []
with open("./Data/dai_viet_su_ki/golden.txt", "r", encoding="utf-8") as f:
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

with open("./Data/Complete/mono_alignments.txt", "w", encoding="utf-8") as f:
    for i in range(len(alignments)):
        f.write(alignments[i][0] + "\t" + alignments[i][1] + "\n")

# Calculate precision, recall and f1 score
match = 0
for alignment in alignments:
    for gold in golden:
        if alignment[0] == gold[0] and alignment[1] == gold[1]:
            match += 1
            break

print("precision", match/len(alignments)) if len(alignments) > 0 else 0
print("recall", match/len(golden)) if len(golden) > 0 else 0
