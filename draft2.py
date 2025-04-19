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

with open("./data/dai_viet_su_ki/trans_src_res.txt", "r", encoding="utf8") as f:
    lines = f.readlines()
    for line in lines:
        parts = line.strip().split('\t')
        sentences.append(parts[1])

with open("./data/dai_viet_su_ki/trans_src_snts.txt", "w", encoding="utf8") as f2:
    for sentence in sentences:
        f2.write(sentence.strip() + '\n')

# with open("./data/dai_viet_su_ki/translation_snts_resplitted.txt", "r", encoding="utf8") as f:
#     lines = f.readlines()
#     with open("./data/dai_viet_su_ki/translation_snts_resplitted.txt", "w", encoding="utf8") as f2:
#         for line in lines:
#             if line.strip() == ".":
#                 continue
#             f2.write(line.strip() + "\n")