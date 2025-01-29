import json
import sys

res = []
def detect_bad_alignment(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for item in data:
            if float(item['similarity']) < 0.3:
                res.append(item)
    tmp_parts = file_path.split('\\')
    # print(tmp_parts)
    dir = '/'.join(tmp_parts[:-1])
    # print(dir)
    with open(dir + '/' + 'bad_alignments.json', 'w', encoding='utf-8') as file:
        json.dump(res, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    detect_bad_alignment(sys.argv[1])