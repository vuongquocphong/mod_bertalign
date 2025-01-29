import bertalign
import numpy as np
from datetime import datetime
from bertalign import model

dirs_list = ["tqdn2"]

def align_dir(dir_name, max_align, model, start_time):
    print(f"Aligning {dir_name} using {model.model_name} model...")
    alignments = []

    tgt = []
    src = []

    with open(f"{dir_name}/chinese_pars.txt", "r", encoding="utf-8") as f:
        pars = f.readlines()
        for par in pars:
            src.append(par.strip())

    with open(f"{dir_name}/translation_pars.txt", "r", encoding="utf-8") as f:
        pars = f.readlines()
        for par in pars:
            tgt.append(par.strip())
    for i in range(len(src)):
        src_text = src[i]
        tgt_text = tgt[i]
        print("Aligning paragraph {}...".format(i + 1))
        aligner = bertalign.Bertalign(src_text, tgt_text, model=model, max_align=max_align)
        aligner.align_sents()
        for bead in (aligner.result):
            src_line = aligner._get_line(bead[0], aligner.src_sents)
            tgt_line = aligner._get_line(bead[1], aligner.tgt_sents)
            # calculate similarity
            alignments.append((src_line, tgt_line))
    end_time = datetime.now()
    time_diff = end_time - start_time
    golden = []
    match = 0
    precision = 0
    recall = 0
    mismatch_list = []
    with open(f"{dir_name}/golden.txt", "r", encoding="utf-8") as f:
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
        # Calculate precision, recall, and F1
        for alignment in alignments:
            is_match = False
            for gold in golden:
                if alignment[0] == gold[0] and alignment[1] == gold[1]:
                    match += 1
                    is_match = True
                    break
            if not is_match:
                mismatch_list.append(alignment)
    with open(f"{dir_name}/alignments.txt", "w", encoding="utf-8") as f:
        for alignment in alignments:
            f.write(alignment[0] + "\t" + alignment[1] + "\n")
    with open(f"{dir_name}/result.txt", "a", encoding="utf-8") as f:
        precision = match / len(alignments) if len(alignments) > 0 else 0
        recall = match / len(golden) if len(golden) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        cur_time = datetime.now().strftime("%H:%M:%S")
        f.write(f"Timestamp: {cur_time}\n")
        f.write(f"Time taken: {time_diff}\n")
        f.write(f"Model: {model.model_name}\n")
        f.write(f"Max align param: {max_align}\n")
        f.write(f"Golden size: {len(golden)}\n")
        f.write(f"Alignments size: {len(alignments)}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1: {f1}\n")
        f.write("--------------------\n")
    with open(f"{dir_name}/mismatch_{model.model_name}.txt", "w", encoding="utf-8") as f:
        for mismatch in mismatch_list:
            f.write(mismatch[0] + "\t" + mismatch[1] + "\n")

if __name__ == "__main__":
    for dir in dirs_list:
        start = datetime.now()
        align_dir(dir, 5, model, start)