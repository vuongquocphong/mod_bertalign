import time
import bertalign
from convert_golden import convert_golden

def get_text_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def check_valid( golden, search_path ):
    result = 0
    for bead in golden:
        src_idx, tgt_idx = bead
        if search_path[src_idx][0] <= tgt_idx <= search_path[src_idx][1]:
            result += 1
    return result / len(golden)

    
source_file = "Data/train/chinese.txt"
target_file = "Data/train/translation.txt"
golden_file = "Data/train/golden.txt"

src_sentences = get_text_from_file(source_file)
tgt_sentences = get_text_from_file(target_file)

golden_index = convert_golden(golden_file)

exit(0)

aligner = bertalign.BertEvaluation(src_sentences, tgt_sentences)

k_start, k_end = 1, 10
w_start, w_end = 1, 50
best_k, best_w = 0, 0
best_result = 0

for k in range(k_start, k_end + 1):
    for w in range(w_start, w_end + 1):
        start_time = time.time()
        window, search_path = aligner.evaluate_k(k, w)
        result = check_valid(golden_index, search_path)
        end_time = time.time()
        if result > best_result:
            best_result = result
            best_k = k
            best_w = w
        
        # Write to file
        with open("evaluate_k.txt", "a", encoding="utf-8") as f:
            f.write(f"k: {k}, w: {w}, window: {window}, result: {result:.4f}, time: {end_time - start_time:.4f}s\n")
            print(f"k: {k}, w: {w}, window: {window}, result: {result:.4f}, time: {end_time - start_time:.4f}s")

print(f"Best k: {best_k}, Best w: {best_w}, Best result: {best_result:.4f}")