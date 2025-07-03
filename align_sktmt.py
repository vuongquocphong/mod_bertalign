# list `the file in sktmt/vi`
import os
from bertalign import Bertalign

files = os.listdir('sktmt/zh')
files = sorted(files)

alignments = []

for file in files:
    zh_par = ""
    vi_par = ""
    with open(f'sktmt/zh/{file}', 'r', encoding='utf-8') as f:
        zh_lines = f.readlines()
        for line in zh_lines:
            zh_par += line + "\n"
    with open(f'sktmt/vi/{file}', 'r', encoding='utf-8') as f:
        vi_lines = f.readlines()
        for line in vi_lines:
            vi_par += line + "\n"
    aligner = Bertalign(src=zh_par, tgt=vi_par, is_split=True)
    aligner.align_sents()

    for bead in (aligner.result):
        src_line = aligner._get_line(bead[0], aligner.src_sents)
        tgt_line = aligner._get_line(bead[1], aligner.tgt_sents)
        # calculate similarity
        alignments.append((src_line, tgt_line))

with open(f'sktmt/alignments.txt', 'w', encoding='utf-8') as f:
    for alignment in alignments:
        f.write(alignment[0] + "\t" + alignment[1] + "\n")
