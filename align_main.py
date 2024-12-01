import bertalign

dirs_list = ["tqdn1", "tqdn2", "tqdn3"]

def align_dir(dir_name):
    alignments = []

    tgt = []
    src = []

    with open(f"{dir_name}/chinese_pars.txt") as f:
        pars = f.readlines()
        for par in pars:
            src.append(par.strip())

    with open(f"{dir_name}/translation_pars.txt") as f:
        pars = f.readlines()
        for par in pars:
            tgt.append(par.strip())

    for i in range(len(src)):
        src_text = src[i]
        tgt_text = tgt[i]
        print("Aligning paragraph {}...".format(i + 1))
        aligner = bertalign.Bertalign(src_text, tgt_text)
        aligner.align_sents()
        for bead in (aligner.result):
            src_line = aligner._get_line(bead[0], aligner.src_sents)
            tgt_line = aligner._get_line(bead[1], aligner.tgt_sents)
            alignments.append((src_line, tgt_line))

    with open(f"{dir_name}/alignments.txt", "w") as f:
        for alignment in alignments:
            f.write(alignment[0] + "\t" + alignment[1] + "\n")

if __name__ == "__main__":
    for dir in dirs_list:
        align_dir(dir)