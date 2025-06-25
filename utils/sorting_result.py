import os

def read_numbers_from_file(filename):
    result = dict()

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            numbers = line.split()
            skip_val, snt_num_pen_val, union_cor_val = float(numbers[0]), float(numbers[1]), float(numbers[2])
            precision_val, recall_val, f1_val = float(numbers[3]), float(numbers[4]), float(numbers[5])
            total_match, total_alignments, total_gold = int(numbers[6]), int(numbers[7]), int(numbers[8])
            result[(skip_val, snt_num_pen_val, union_cor_val)] = (
                precision_val, recall_val, f1_val,
                total_match, total_alignments, total_gold
            )
    
    # Sort the result by F1 then by skip value, snt_num_pen, and union_cor
    sorted_result = sorted(result.items(), key=lambda x: (-x[1][2], -x[0][0], x[0][1], x[0][2]))
    return sorted_result

def write_sorted_results_to_file(sorted_results, output_filename):
    with open(output_filename, 'w') as f:
        # f.write(f"{'Skip':<8}{'SntNumPen':<12}{'UnionScore':<12}"
        #         f"{'Precision':<10}{'Recall':<10}{'F1':<10}"
        #         f"{'Match':<8}{'Alignments':<12}{'Gold':<10}\n")
        for key, values in sorted_results:
            skip_val, snt_num_pen_val, union_cor_val = key
            precision, recall, f1, total_match, total_alignments, total_gold = values
            f.write(f"{skip_val:<8}{snt_num_pen_val:<12}{union_cor_val:<12}"
                    f"{precision:<10.4f}{recall:<10.4f}{f1:<10.4f}"
                    f"{total_match:<8}{total_alignments:<12}{total_gold:<10}\n")

if __name__ == "__main__":
    
    # Read all the file from the specified directory
    directory = "../Evaluation_results/Evaluation"
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            result = read_numbers_from_file(file_path)
            write_sorted_results_to_file(result, file_path)
    