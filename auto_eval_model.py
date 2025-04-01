from bertalign.encoder import Encoder
from align_main import * 
from eval_alignments import eval

# list_model = ["laser2", "intfloat/multilingual-e5-large", "jinaai/jina-embeddings-v3"]
# list_model = ["distiluse-base-multilingual-cased-v2"]
# list_model = ["Alibaba-NLP/gte-modernbert-base"]
# list_model = ["sentence-transformers/clip-ViT-B-32-multilingual-v1"]
# list_model = ["Alibaba-NLP/gte-multilingual-base"]
# list_model = ["sentence-transformers/all-MiniLM-L6-v2"]
list_model = ["sentence-transformers/sentence-t5-large"]

def auto_eval_model(dir_name, num_overlaps, top_k):
    ner_dict = load_ner_json_to_dict("cn_sv_dict.json")
    # print(ner_dict)
    for model_name in list_model:
        model = Encoder(model_name)
        align_dir(dir_name, top_k, num_overlaps, model, datetime.now(), ner_dict=ner_dict)
        # eval(dir_name, num_overlaps, model)
        print(f"Done evaluating {dir_name} using {model_name} model.")
        print("=========================================")

if __name__ == "__main__":
    import sys
    dir_name = sys.argv[1]
    num_overlaps = int(sys.argv[2])
    top_k = int(sys.argv[3])
    auto_eval_model(dir_name, num_overlaps, top_k)