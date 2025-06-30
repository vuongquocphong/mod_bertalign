import torch
import bertalign

def load_data_par(file_path):
    pars = []
    with open(file_path, "r", encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) > 0:
                pars.append(line)
    return pars

def load_gold(file_path):
    golden = []
    with open(file_path, "r", encoding="utf-8") as f:
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
    return golden

def continuous_numbers(start, end, step=1.0):
    """
    Returns a list of continuous numbers from start to end (exclusive), with the given step.
    All numbers are rounded to 2 decimal places.
    """
    numbers = []
    current = start
    if step == 0: raise ValueError("Step cannot be zero.")
    if start > end: raise ValueError("Start must be less than or equal to end.")
    
    def norm_zero(x):
        return 0.0 if abs(x) < 1e-8 else x

    while current <= end:
        val = norm_zero(round(current, 2))
        numbers.append(val)
        current += step
    
    if len(numbers) == 0 or numbers[-1] != norm_zero(round(end, 2)):
        numbers.append(norm_zero(round(end, 2)))

    return numbers

def evaluate(alignments, golden):
    match, golden_length, golden_current = 0, len(golden), 0 
    for alignment in alignments:
        for index in range(golden_current, golden_length):
            if alignment[0] == golden[index][0] and alignment[1] == golden[index][1]:
                match += 1
                golden_current = index + 1
                break
    precision = match / len(alignments) if len(alignments) > 0 else 0
    recall = match / len(golden) if len(golden) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1, match, len(alignments), len(golden)

# Define the parameters for the alignment
skip_cost = continuous_numbers( -1.0, 0.0, 0.01 )
# snt_num_penalty = continuous_numbers( 0.0, 0.5, 0.01 )
# union_score = continuous_numbers( 0.0, 1.0, 0.05 )
snt_num_penalty = continuous_numbers( 0.0, 0.0, 0.01 )
union_score = continuous_numbers( 0.0, 0.0, 0.05 )
# model = bertalign.Encoder("LaBSE")

def run_alignment_par(src_pars, tgt_pars, golden_par, name):
    aligner = bertalign.BertalignModified(src=src_pars, tgt=tgt_pars)

    for skip_val in skip_cost:
        for snt_num_pen_val in snt_num_penalty:
            for union_cor_val in union_score:
                # print(f"Aligning with skip: {skip_val}, sentence_num_penalty: {snt_num_pen_val}, union_score: {union_cor_val}")
                aligner.align_sents(input_skip=skip_val, input_union_score=union_cor_val, input_sentence_num_penalty=snt_num_pen_val)
                alignments = []
                for bead in aligner.result:
                    src_line = aligner._get_line(bead[0], aligner.src_sents)
                    tgt_line = aligner._get_line(bead[1], aligner.tgt_sents, ' ')
                    alignments.append((src_line, tgt_line))
                precision, recall, f1, match, total_align, total_gold = evaluate(alignments, golden_par)
                # print(f"Results - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                # Save the results to a file
                with open(f"Evaluation_results/Origin_method/newRegEx/{name}.txt", "a", encoding="utf-8") as f:
                    f.write(
                        f"{skip_val:<8}{snt_num_pen_val:<12}{union_cor_val:<12}"
                        f"{precision:<10.4f}{recall:<10.4f}{f1:<10.4f}"
                        f"{match:<8}{total_align:<12}{total_gold:<10}\n"
                    )
    return None
    # return alignments

# Load the data
# src_path = "/home/hoktro/mod_bertalign/Data/tam_quoc_dien_nghia/tqdn1/chinese_pars.txt"
# tgt_path = "/home/hoktro/mod_bertalign/Data/tam_quoc_dien_nghia/tqdn1/translation_pars.txt"
# gold_path = "/home/hoktro/mod_bertalign/Data/tam_quoc_dien_nghia/tqdn1/golden.txt"
# prename = "TQDN1"
# golden_end = [1, 8, 15, 29, 47, 51, 63, 72, 82, 93, 109, 117, 124, 128, 132, 136, 164, 184, 191, 192, 200, 214, 217, 259, 278, 293, 312, 318, 331, 339, 374, 409, 439, 440, 446, 447, 461, 468, 479, 486, 503, 529, 543, 560, 570, 635, 660, 661, 673, 684, 690, 695, 700, 727, 742, 770, 786, 799, 819, 829, 848, 854, 855, 869, 878, 901, 937, 950, 954, 962, 983, 1043, 1044, 1060, 1069, 1109, 1110, 1115, 1153, 1156, 1162, 1169, 1215, 1234, 1255, 1270, 1290, 1291, 1292, 1301, 1313, 1318, 1325, 1365, 1373, 1381, 1396, 1422, 1442, 1458, 1473, 1484, 1490, 1504, 1505, 1511, 1513, 1531, 1562, 1637, 1663, 1678, 1699, 1711, 1712, 1736, 1744, 1795, 1811, 1827, 1843, 1849, 1857, 1909, 1922, 1929, 1950, 1955, 1956, 1970, 1986, 2009, 2052, 2065, 2099, 2113, 2120]

# src_path = "/home/hoktro/mod_bertalign/Data/dai_nam_chinh_bien_liet_truyen/chinese_pars.txt"
# tgt_path = "/home/hoktro/mod_bertalign/Data/dai_nam_chinh_bien_liet_truyen/translation_pars.txt"
# gold_path = "/home/hoktro/mod_bertalign/Data/dai_nam_chinh_bien_liet_truyen/golden.txt"
# prename = "DaiNamChinhBienLietTruyen"
# golden_end = [16, 39, 52, 66, 77, 99, 117, 146, 172, 190, 213, 232, 239, 262, 299, 318, 341, 367, 387, 410, 421, 433]

src_path = "/home/hoktro/mod_bertalign/Data/dai_viet_su_ki/train_src_pars.txt"
tgt_path = "/home/hoktro/mod_bertalign/Data/dai_viet_su_ki/train_tgt_pars.txt"
gold_path = "/home/hoktro/mod_bertalign/Data/dai_viet_su_ki/train_gold.txt"
prename = "DVSK"
golden_end = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000, 5200, 5400, 5600, 5800, 6000]


src_pars = load_data_par(src_path)
tgt_pars = load_data_par(tgt_path)
golden_pars = load_gold(gold_path)


if __name__ == "__main__":
    
    # Check if the number of source and target paragraphs match
    if len(src_pars) != len(tgt_pars):
        raise ValueError("The number of source paragraphs does not match the number of target paragraphs.")
    
    start_paragraph = 1
    bathch_size = 137
    
    # Align each paragraph pair
    total_paragraphs = len(src_pars)
    for i in range(start_paragraph - 1, min(total_paragraphs, start_paragraph + bathch_size - 1)):
        print(f"Aligning paragraph {i + 1}/{len(src_pars)}")
        src_paragraph = src_pars[i]
        tgt_paragraph = tgt_pars[i]
        golden_paragraph = golden_pars[0: golden_end[0]] if i == 0 else golden_pars[golden_end[i - 1]: golden_end[i]]
        alignments = run_alignment_par(src_paragraph, tgt_paragraph, golden_paragraph, f"{prename}_{i + 1}")
        torch.cuda.empty_cache()
        # print(f"Alignments for paragraph {i + 1}: {alignments}")
    
    print("Alignment process completed.")

    # for i in range(total_paragraphs):
    #     print(f"Currently processing paragraph {i + 1}/{total_paragraphs}")
    #     print(f"First 10 characters of source: {src_pars[i][:10]}")
    #     print(f"First 10 characters of target: {tgt_pars[i][:10]}")
    #     golden_paragraph = golden_pars[0: golden_end[0]] if i == 0 else golden_pars[golden_end[i - 1]: golden_end[i]]
    #     print(f"First 10 characters of golden: {golden_paragraph[0][0][:10] if golden_paragraph else 'N/A'}")