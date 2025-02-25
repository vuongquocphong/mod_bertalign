from bertalign.encoder import Encoder
import numpy as np
import json
from underthesea import word_tokenize
from bertalign import model

zh_marks = {'。', '！', '？', '；', '…', '：', '，', '、', '「', '」', '（', '）', '《', '》', '—', '～', '‘', '’', '“', '”', '［', '］', '【', '】', '〈', '〉', '﹏', '＿', '＠', '＃', '＆', '＊', '＋', '－', '／', '＝', '＜', '＞', '＼', '｜', '＾', '｀', '｛', '｝', '～', '！', '＠', '＃', '＄', '％', '＾', '＆', '＊', '（', '）', '＿', '＋', '－', '＝', '｜', '｛', '｝', '【', '】', '‘', '’', '“', '”', '；', '：', '？', '，', '、', '。', '《', '》', '〈', '〉', '…', '—', '～', '﹏', '＼', '／', '＜', '＞', '％', '＃', '＆', '＠'}

def eval(dir_name, num_overlaps, model):
    print(f"Generating eval file using {model.model_name} model...")
    def get_similarity(sent1, sent2, num_overlaps):
        em1, em1_len = model.transform([sent1], num_overlaps)
        em2, em2_len = model.transform([sent2], num_overlaps)
        em1_vec = em1[0][0]
        em2_vec = em2[0][0]
        return np.dot(em1_vec, em2_vec) / (np.linalg.norm(em1_vec) * np.linalg.norm(em2_vec))
    with open(f"{dir_name}/alignments.txt", "r", encoding="utf-8") as f:
        alignments = f.readlines()
        for i in range(len(alignments)):
            first_part = ""
            second_part = ""
            for char in alignments[i]:
                if char == "\t":
                    second_part = alignments[i][alignments[i].index(char) + 1:]
                    break
                first_part += char
            alignments[i] = (first_part, second_part)
    sim = 0
    weighted_sim = 0
    sum_weight = 0
    results = []
    for alignment in alignments:
        print(f"Calculating similarity for {alignment[0]} and {alignment[1]}...")
        json_data = {}
        json_data['source'] = alignment[0]
        json_data['target'] = alignment[1]
        tmp = get_similarity(alignment[0], alignment[1], num_overlaps)
        tmp_zh_len = sum([1 for char in alignment[0] if char not in zh_marks])
        tmp_vi_len = len(word_tokenize(alignment[1]))
        tmp_weight = tmp_zh_len + tmp_vi_len
        json_data['similarity'] = str(tmp)
        json_data['weight'] = str(tmp_weight)
        sim += tmp
        weighted_sim += tmp * tmp_weight
        sum_weight += tmp_weight
        results.append(json_data)
    
    with open(f"{dir_name}/eval_results_{model.model_name}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        f.write("\n")
        f.write(f"Average similarity: {sim / len(alignments)}")
        f.write("\n")
        f.write(f"Weighted average similarity: {weighted_sim / sum_weight}")
        f.write("\n")

if __name__ == "__main__":
    eval("tqdn3", 5, model)