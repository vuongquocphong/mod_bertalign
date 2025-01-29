from bertalign.encoder import Encoder
from align_main import * 
from eval_alignments import eval

list_model = ["LaBSE", "all-MPNet-base-v2", "paraphrase-multilingual-mpnet-base-v2", "distiluse-base-multilingual-cased-v2"]

def auto_eval_model(dir_name, num_overlaps):
    for model_name in list_model:
        model = Encoder(model_name)
        align_dir(dir_name, num_overlaps, model, datetime.now())
        eval(dir_name, num_overlaps, model)
        print(f"Done evaluating {dir_name} using {model_name} model.")
        print("=========================================")

if __name__ == "__main__":
    import sys
    dir_name = sys.argv[1]
    num_overlaps = int(sys.argv[2])
    auto_eval_model(dir_name, num_overlaps)