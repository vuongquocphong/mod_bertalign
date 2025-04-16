import sys
import os

cols = ["viet", "chinese", "eval"]

# INPUT_FILE = str(sys.argv[1])
# OUTPUT_FILE = str(sys.argv[2])

def extract_data(file):
    # open xlsx file
    import pandas as pd
    data = pd.read_excel(file)
    viet = data[cols[0]].tolist()
    chinese = data[cols[1]].tolist()
    eval = data[cols[2]].tolist()
    res_data = []
    for i in range(len(eval)):
        if eval[i] != 'bad':
            res_data.append((viet[i], chinese[i]))
    # turn the data into a tsv file
    with open("OUTPUT_FILE", 'a', encoding='utf8') as f:
        for item in res_data:
            f.write("%s\t%s\n" % (item[1], item[0]))

from transformers import pipeline

def trans_test():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    translator = pipeline(
        "translation",
        model=model_name,
        tokenizer=model_name,
        src_lang="zh_CN",
        tgt_lang="vi_VN",
    )

    test_set = [
        "朝列大夫國子監司業兼史官修撰臣吳士連編。",
        "按黄帝時建萬國以交趾界於西南遠在百粤之表。",
        "堯命𦏁氏宅南交定南方交趾之地。"
    ]

    for text in test_set:
        result = translator(text)
        print(result[0]["translation_text"])

if __name__ == "__main__":
    trans_test()