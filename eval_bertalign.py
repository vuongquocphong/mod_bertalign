import bertalign
from bertalign import Encoder

src_par = ""
tgt_par = ""

with open("./data/dai_viet_su_ki/chinese_snts.txt", "r", encoding="utf8") as f:
    lines = f.readlines()
    for line in lines:
        src_par += line.strip() + "\n"

with open("./data/dai_viet_su_ki/translation_snts.txt", "r", encoding="utf8") as f:
    lines = f.readlines()
    for line in lines:
        tgt_par += line.strip() + "\n"

model = Encoder("LaBSE")
aligner = bertalign.Bertalign(src=src_par, tgt=tgt_par, is_split=True, model=model)

aligner.align_sents()

alignments = []
for bead in (aligner.result):
    src_line = aligner._get_line(bead[0], aligner.src_sents)
    tgt_line = aligner._get_line(bead[1], aligner.tgt_sents)
    # calculate similarity
    alignments.append((src_line, tgt_line))

golden = []
with open("./data/dai_viet_su_ki/golden.txt", "r", encoding="utf-8") as f:
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

print("precision", match/len(alignments)) if len(alignments) > 0 else 0
print("recall", match/len(golden)) if len(golden) > 0 else 0