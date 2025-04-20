# from underthesea import sent_tokenize

# with open("./data/dai_viet_su_ki/translation_snts.txt", "r", encoding="utf8") as f:
#     lines = f.readlines()
#     par = ""
#     for line in lines:
#         par += line.strip() + " "
#     sentences = sent_tokenize(par)
#     with open("./data/dai_viet_su_ki/translation_snts_resplitted.txt", "w", encoding="utf8") as f2:
#         for sentence in sentences:
#             f2.write(sentence.strip() + '\n')

# from underthesea import sent_tokenize

sentences = []
golden = []

# with open("./data/dai_viet_su_ki/trans_src_res.txt", "r", encoding="utf8") as f:
#     lines = f.readlines()
#     for line in lines:
#         parts = line.strip().lower().split('\t')
#         sentences.append(parts[1])

# with open("./data/dai_viet_su_ki/trans_src_snts.txt", "w", encoding="utf8") as f2:
#     for sentence in sentences:
#         f2.write(sentence.strip() + '\n')

# with open("./data/dai_viet_su_ki/deepseek_trans_src_snts.txt", "r", encoding="utf8") as f:
#     lines = f.readlines()
#     with open("./data/dai_viet_su_ki/deepseek_trans_src_snts.txt", "w", encoding="utf8") as f2:
#         for line in lines:
#             f2.write(line.strip().lower() + "\n")
from bertalign import Encoder
import numpy as np

model = Encoder("LaBSE")

def get_similarity(sent1, sent2, num_overlaps):
    em1, em1_len = model.transform([sent1], num_overlaps)
    em2, em2_len = model.transform([sent2], num_overlaps)
    em1_vec = em1[0][0]
    em2_vec = em2[0][0]
    return np.dot(em1_vec, em2_vec) / (np.linalg.norm(em1_vec) * np.linalg.norm(em2_vec))

trans_snts = []
ref_snts = []

with open("./data/dai_viet_su_ki/deepseek_trans_src_snts.txt", "r", encoding="utf8") as f:
    lines = f.readlines()
    for line in lines:
        trans_snts.append(line.strip())

with open("./data/dai_viet_su_ki/translation_snts.txt", "r", encoding="utf8") as f:
    lines = f.readlines()
    for line in lines:
        ref_snts.append(line.strip())

with open("./data/dai_viet_su_ki/tmp.txt", "w") as f:
    for i in range(len(trans_snts)):
        print(f"Processing sentence pair {i}")
        sim = get_similarity(trans_snts[i], ref_snts[i], 4)
        f.write(str(sim) + "\n")