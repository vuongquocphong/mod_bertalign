import json
import os
import pandas as pd

# read excel file and extract the "chinese" and "NER" columns

def read_excel(folder_path, output_file_path):
    list_res = []
    # Read the Excel file
    list_files = os.listdir(folder_path)
    # Check if the folder contains any Excel files
    if not any(file.endswith('.xlsx') for file in list_files):
        raise ValueError("No Excel files found in the specified folder.")
    for file in list_files:
        if file.endswith('.xlsx'):
            excel_file_path = os.path.join(folder_path, file)
            df = pd.read_excel(excel_file_path)
            # Check if the required columns exist
            if 'chinese' in df.columns and 'NER' in df.columns:
                chinese_sentences = df['chinese'].tolist()
                ner = df['NER'].tolist()
                for sentence, ner_list in zip(chinese_sentences, ner):
                    splitted_sentences = sentence.split("\n")
                    sentence = [s.strip() for s in splitted_sentences if s.strip()]
                    sentence = "".join(sentence).replace(".", "。")
                    print(type(ner_list))
                    # The ner_list is in the following format (it is a string):
                    #{
                    #   'PER': ['陶󰁮', '克昌'],
                    #   'LOC': [],
                    #   'ORG': ['翰林院'],
                    #   'TITLE': ['正使', '郎中'],
                    #   'TIME': ['六月初七日', '十日', '十六日', '八月 初六日', '七月二十七日']
                    # }
                    # Convert the ner_list to a list of entities
                    # Convert the string representation of the dictionary to an actual dictionary
                    ner_list = eval(ner_list)
                    # Extract entities from the dictionary
                    ner_extracted_list = []
                    for entity_type, entities in ner_list.items():
                        if entities:
                            for entity in entities:
                                ner_extracted_list.append(entity)
                    print(ner_list)
                    list_res.append({"sentence": sentence, "NER": ner_list})
            else:
                raise ValueError("Required columns 'chinese' and 'NER' not found in the Excel file.")
            break
    # Write the extracted data to a JSON file
    if list_res:
        with open(output_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(list_res, json_file, ensure_ascii=False, indent=4)
    else:
        raise ValueError("No Excel files found in the specified folder.")

if __name__ == "__main__":
    import sys
    folder_path = sys.argv[1]  # Path to the folder containing the Excel file
    output_file_path = sys.argv[2]  # Path to the output JSON file
    # Check if the folder exists
    read_excel(folder_path, output_file_path)
    print(f"NER data has been written to {output_file_path}")
