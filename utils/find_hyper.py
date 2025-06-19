import os

def get_txt_filenames_and_first_lines(folder_path):
    """
    Returns a list of tuples: (file_name, first_line) for each .txt file in the folder.
    """
    result = []
    for f in os.listdir(folder_path):
        if f.endswith('.txt') and os.path.isfile(os.path.join(folder_path, f)):
            file_path = os.path.join(folder_path, f)
            with open(file_path, 'r', encoding='utf-8') as file:
                first_line = file.readline().rstrip('\n')
            
            nums = first_line.split()
            result.append((f, nums[-1]))
    
    # Sort by the file name
    result.sort(key=lambda x: x[0])
    
    return result

def write_results_to_file(results, output_file):
    """
    Writes the results to a specified output file.
    """
    # Write results to the output file, each line contains the file name and its first line
    with open(output_file, 'w', encoding='utf-8') as f:
        for file_name, golden in results:
            # Write file name and first line, separated by a tab for clarity
            f.write(f"{file_name:<30} {golden}\n")

if __name__ == "__main__":
    folder_path = '../Evaluation_results/Training'
    output_file = 'files.txt'
    
    results = get_txt_filenames_and_first_lines(folder_path)
    write_results_to_file(results, output_file)
    
    print(f"Results written to {output_file}")