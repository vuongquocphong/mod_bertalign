import os


combine_result = dict()
final_result = dict()

global_golden = 0

def read_numbers_from_file(filename):

    gol = -1
    numbers_of_line = 0

    with open(filename, 'r') as file:
        for line in file:
            numbers_of_line += 1
            line = line.strip()
            numbers = line.split()
            skip_val, snt_num_pen_val, union_cor_val = numbers[0], numbers[1], numbers[2]
            total_match, total_alignments, total_gold = numbers[6], numbers[7], numbers[8]

            if gol == -1:
                gol = int(total_gold)
            elif gol != int(total_gold):
                print(f"Warning: Total gold count mismatch in file {filename}. Expected {gol}, found {total_gold}.")
                raise ValueError("Total gold count mismatch.")

            if (skip_val, snt_num_pen_val, union_cor_val) not in combine_result:
                combine_result[(skip_val, snt_num_pen_val, union_cor_val)] = [int(total_match), int(total_alignments), int(total_gold)]
            else:
                combine_result[(skip_val, snt_num_pen_val, union_cor_val)][0] += int(total_match)
                combine_result[(skip_val, snt_num_pen_val, union_cor_val)][1] += int(total_alignments)
                combine_result[(skip_val, snt_num_pen_val, union_cor_val)][2] += int(total_gold)
    
    if numbers_of_line != 26691:
        print(f"Warning: File {filename} does not contain 26691 lines. It contains {numbers_of_line} lines.")

    global global_golden
    global_golden += gol

    # print(f"Processed file: {filename}, Total gold count: {gol}, Global gold count: {global_golden}")

def calculate_scores():
    for key, values in combine_result.items():
        
        total_match, total_alignments, total_gold = values[0], values[1], values[2]

        precision = total_match / total_alignments if total_alignments > 0 else 0
        recall = total_match / total_gold if total_gold > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        final_result[key] = (precision, recall, f1, total_match, total_alignments, total_gold)

if __name__ == "__main__":
    
    # Read all the file from the specified directory
    directory = "/home/hoktro/mod_bertalign/Evaluation_results/Evaluation"

    files = os.listdir(directory)
    files = [f for f in files if f.endswith('.txt')]
    files.sort()

    for filename in files:
        file_path = os.path.join(directory, filename)
        read_numbers_from_file(file_path)
    
    # Calculate scores
    calculate_scores()
    
    # Write the final results to a file and align for better readability
    with open( directory + "/.final_result.re", "w") as f:
        f.write(f"{'Skip':<8}{'SntNumPen':<12}{'UnionScore':<12}"
                f"{'Precision':<10}{'Recall':<10}{'F1':<10}"
                f"{'Match':<8}{'Alignments':<12}{'Gold':<10}\n")
        for key, values in final_result.items():
            skip_val, snt_num_pen_val, union_cor_val = key
            precision, recall, f1, total_match, total_alignments, total_gold = values
            f.write(f"{skip_val:<8}{snt_num_pen_val:<12}{union_cor_val:<12}"
                    f"{precision:<10.4f}{recall:<10.4f}{f1:<10.4f}"
                    f"{total_match:<8}{total_alignments:<12}{total_gold:<10}\n")
            
    # Sort the final results by F1 score in descending order
    sorted_results = sorted(final_result.items(), key=lambda x: (-x[1][2], -float(x[0][0]), float(x[0][1]), float(x[0][2])))

    # Write the sorted results to a file
    with open( directory + "/.sorted_final_results.re", "w") as f:
        f.write(f"{'Skip':<8}{'SntNumPen':<12}{'UnionScore':<12}"
                f"{'Precision':<10}{'Recall':<10}{'F1':<10}"
                f"{'Match':<8}{'Alignments':<12}{'Gold':<10}\n")
        for key, values in sorted_results:
            skip_val, snt_num_pen_val, union_cor_val = key
            precision, recall, f1, total_match, total_alignments, total_gold = values
            if float(skip_val) < -0.5: continue
            f.write(f"{skip_val:<8}{snt_num_pen_val:<12}{union_cor_val:<12}"
                    f"{precision:<10.4f}{recall:<10.4f}{f1:<10.4f}"
                    f"{total_match:<8}{total_alignments:<12}{total_gold:<10}\n")
    