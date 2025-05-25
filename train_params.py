import optuna
import bertalign_modified
# from bertalign_modified.utils import load_nom_dict

def load_data(file_path):
    snts = """"""
    with open(file_path, "r", encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            snts += line.strip() + "\n"
    return snts

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

def run_alignment(src_snts, tgt_snts, nom_dict_path, skip, snt_num_pen_val, union_cor_val):
    model = bertalign_modified.Encoder("LaBSE")
    alignments = []
    aligner = bertalign_modified.Bertalign(src=src_snts, tgt=tgt_snts, model=model, is_split=True, skip=skip, snt_num_pen_val=snt_num_pen_val, union_cor_val=union_cor_val, nom_dict_path=nom_dict_path)
    aligner.align_sents()
    for bead in aligner.result:
        src_line = aligner._get_line(bead[0], aligner.src_sents)
        tgt_line = aligner._get_line(bead[1], aligner.tgt_sents)
        alignments.append((src_line, tgt_line))
    return alignments

def run_alignment_par(src_pars, tgt_pars, nom_dict_path, skip, snt_num_pen_val, union_cor_val):
    model = bertalign_modified.Encoder("LaBSE")
    alignments = []
    for i in range(len(src_pars)):
        print(f"Processing paragraph {i + 1}/{len(src_pars)}")
        src_snts = src_pars[i]
        tgt_snts = tgt_pars[i]
        aligner = bertalign_modified.Bertalign(src=src_snts, tgt=tgt_snts, model=model, skip=skip, snt_num_pen_val=snt_num_pen_val, union_cor_val=union_cor_val, nom_dict_path=nom_dict_path)
        aligner.align_sents()
        for bead in aligner.result:
            src_line = aligner._get_line(bead[0], aligner.src_sents)
            tgt_line = aligner._get_line(bead[1], aligner.tgt_sents)
            alignments.append((src_line, tgt_line))
    return alignments

def evaluate(alignments, golden):
    match = 0 
    for alignment in alignments:
        for gold in golden:
            if alignment[0] == gold[0] and alignment[1] == gold[1]:
                match += 1
                break
    precision = match / len(alignments) if len(alignments) > 0 else 0
    recall = match / len(golden) if len(golden) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def objective(trial):
    src_path = "/home/zrefalogia/mod_bertalign/data/Complete/chinese_val_pars.txt"
    tgt_path = "/home/zrefalogia/mod_bertalign/data/Complete/translation_val_pars.txt"
    gold_path = "/home/zrefalogia/mod_bertalign/data/Complete/real_golden.txt"
    nom_dict_path = "/home/zrefalogia/mod_bertalign/data/dictionary/D_203_single_char_nom_qn_dictionary_thi_vien.xlsx"
    
    skip = trial.suggest_float("skip", -0.5, -0.12746239864942988)
    snt_num_pen_val = trial.suggest_float("snt_num_pen", 0.0036033282267109398, 0.06)
    union_cor_val = trial.suggest_float("union_cor_val", 0.17660243143178916, 0.6)

    src_pars = load_data_par(src_path)
    tgt_pars = load_data_par(tgt_path)

    alignments = run_alignment_par(src_pars, tgt_pars, nom_dict_path, skip, snt_num_pen_val, union_cor_val)

    golden = load_gold(gold_path)
    
    precision, _, _ = evaluate(alignments, golden)

    return precision  # Optimize precision

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")  # Set to maximize
    study.optimize(objective, n_trials=10)
    
    print("Best parameters: ", study.best_params)
    print("Best Precision score: ", study.best_value)

    # If you need to use the dictionary outside optimization
    # ner_dict_path = "/home/zrefalogia/mod_bertalign/data/dictionary/D_203_single_char_nom_qn_dictionary_thi_vien.xlsx"
    # ner_dict = load_nom_dict(ner_dict_path)
    # print(ner_dict)
