import os


general_ranking = dict()

def read_numbers_from_file(filename):

    current_ranking = set()
    current_stored = dict()

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            numbers = line.split()
            skip_val, snt_num_pen_val, union_cor_val = float(numbers[0]), float(numbers[1]), float(numbers[2])
            precision, recall, f1 = float(numbers[3]), float(numbers[4]), float(numbers[5])

            if skip_val < -0.5: continue

            current_ranking.add((precision, recall, f1))
            current_stored[(skip_val, snt_num_pen_val, union_cor_val)] = (precision, recall, f1)
    
    # Sort the current ranking
    sorted_ranking = sorted(current_ranking, key=lambda x: (-x[2], -x[0], -x[1]))

    # Update the global ranking ( the sum of rank of coefficients tuple in all files )
    for key in current_stored:
        current_rank = sorted_ranking.index(current_stored[key]) + 1
        if key not in general_ranking: general_ranking[key] = 0
        general_ranking[key] += current_rank

if __name__ == "__main__":
    # Read all the file from the specified directory
    directory = "/home/hoktro/mod_bertalign/Evaluation_results/Evaluation"
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            read_numbers_from_file(file_path)

    # Sort the general ranking by the sum of ranks in increasing order then by skip, snt_num_pen, and union_cor values
    sorted_general_ranking = sorted(general_ranking.items(), 
                                            key=lambda x: (x[1], -float(x[0][0]), float(x[0][1]), float(x[0][2])))

    # Write the final results to a file and align for better readability
    with open(directory + "/.general_ranking.re", "w") as f:
        f.write(f"{'Skip':<8}{'SntNumPen':<12}{'UnionScore':<12}"
                f"{'Rank':<10}\n")
        for key, rank in sorted_general_ranking:
            skip_val, snt_num_pen_val, union_cor_val = key
            f.write(f"{skip_val:<8}{snt_num_pen_val:<12}{union_cor_val:<12}"
                    f"{rank:<10}\n")