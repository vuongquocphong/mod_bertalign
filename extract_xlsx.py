# read "DichNghia.txt"

sentences = []

with open("./DaiNamChinhBienLietTruyen/DichNghia.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    # remove leading/trailing whitespaces, endline characters
    lines = [line.strip() for line in lines]
    # combine all lines into one string
    text = " ".join(lines)
    # split the text into sentences
    from underthesea import sent_tokenize
    sentences = sent_tokenize(text)

# Write the sentences to a new file
with open("./DaiNamChinhBienLietTruyen/DichNghia_sentences.txt", "w", encoding="utf-8") as f:
    for sentence in sentences:
        f.write(sentence + "\n")
