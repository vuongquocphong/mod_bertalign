from bertalign import model
import json
import numpy as np

def grade(dir_name):
    def get_similarity(sent1, sent2, num_overlaps):
        em1, em1_len = model.transform([sent1], num_overlaps)
        em2, em2_len = model.transform([sent2], num_overlaps)
        em1_vec = em1[0][0]
        em2_vec = em2[0][0]
        return np.dot(em1_vec, em2_vec) / (np.linalg.norm(em1_vec) * np.linalg.norm(em2_vec))
    # read chinese.txt
    chinese = []
    with open(dir_name + '/chinese.txt', 'r', encoding='utf-8') as f:
        for line in f:
            chinese.append(line.strip())
    # read translation.txt
    translation = []
    with open(dir_name + '/translation.txt', 'r', encoding='utf-8') as f:
        for line in f:
            translation.append(line.strip())
    results = []
    sim_list = []
    min_sim = 1
    max_sim = 0
    for i in range(len(chinese)):
        if chinese[i] == translation[i]:
            continue
        print(f"Calculating similarity for {chinese[i]} and {translation[i]}...")
        json_data = {}
        json_data['source'] = chinese[i]
        json_data['target'] = translation[i]
        tmp = get_similarity(chinese[i], translation[i], 5)
        json_data['similarity'] = str(tmp)
        sim_list.append(tmp)
        if tmp < min_sim:
            min_sim = tmp
        if tmp > max_sim:
            max_sim = tmp
        results.append(json_data)
    total_sim = sum(sim_list)
    passed_percentage = len([sim for sim in sim_list if sim >= 0.19]) / len(sim_list)
    with open(f"{dir_name}/grade_results_{model.model_name}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        f.write("\n")
        f.write(f"Number of sentences: {len(chinese)}")
        f.write("\n")
        f.write(f"Average similarity: {total_sim / len(chinese)}")
        f.write("\n")
        f.write(f"Minimum similarity: {min_sim}")
        f.write("\n")
        f.write(f"Maximum similarity: {max_sim}")
        f.write("\n")
        f.write(f"Percentage of sentences passed (>0.15): {passed_percentage}")
        f.write("\n")

if __name__ == "__main__":
    import sys
    grade(sys.argv[1])