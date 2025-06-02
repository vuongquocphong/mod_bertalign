from bertalign import Bertalign

source_file = "./tdk_chinese_pars.txt"
target_file = "./tdk_translation_pars.txt"

src_pars = []
tgt_pars = []
with open(source_file, "r", encoding="utf-8") as f:
    source_snts = f.readlines()
    for snt in source_snts:
        src_pars.append(snt.strip())

with open(target_file, "r", encoding="utf-8") as f:
    target_snts = f.readlines()
    for snt in target_snts:
        tgt_pars.append(snt.strip())

alignments = []
for i in range(len(src_pars)):
    aligner = Bertalign(src=src_pars[i], tgt=tgt_pars[i], is_split=False)
    aligner.align_sents()
    for bead in aligner.result:
        src_line = aligner._get_line(bead[0], aligner.src_sents)
        tgt_line = aligner._get_line(bead[1], aligner.tgt_sents)
        # calculate similarity
        alignments.append((src_line, tgt_line))

with open("./tdk_alignments.txt", "w", encoding="utf-8") as f:
    for alignment in alignments:
        f.write(alignment[0] + "\t" + alignment[1] + "\n")