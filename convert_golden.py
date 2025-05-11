import re
from underthesea import sent_tokenize

def _split_zh(text, limit=1000):
	sent_list = []
	text = re.sub('(?P<quotation_mark>([。.？！](?![”’"」\'）])))', r'\g<quotation_mark>\n', text)
	text = re.sub('(?P<quotation_mark>([。.？！]|…{1,2})[”’"」\'）])', r'\g<quotation_mark>\n', text)
	sent_list_ori = text.splitlines()
	for sent in sent_list_ori:
		sent = sent.strip()
		if not sent:
			continue
		else:
			while len(sent) > limit:
				temp = sent[0:limit]
				sent_list.append(temp)
				sent = sent[limit:]
			sent_list.append(sent)
	return sent_list

def _split_vn(text):
	sents = sent_tokenize(text)
	sents = [sent.strip() for sent in sents]
	sents = [sent for sent in sents if _clean_vn(sent)]
	return sents

def _clean_vn(text):
	# Remove " and . of the text
	text = re.sub(r'\"', '', text)
	text = re.sub(r'\.', '', text)

	return text

def convert_golden(file_path, enable_debug = False):
	# Read golden text files
	data = []
	with open(file_path, 'r', encoding="utf-8") as f:
		data = f.readlines()

	debug_zh = []
	debug_vn = []

	index_src, index_tgt = 1, 1
	index_golden = []

	for line in data:
		line = line.split("\t")

		if len(line) != 2:
			print("Error: The number of columns in the line is not equal to 2.")
			print("Line: ", line)
			continue
		
		zh_text, vn_text = line[0].strip(), line[1].strip()
		zh_text = _split_zh(zh_text)
		vn_text = _split_vn(vn_text)

		debug_zh.extend(zh_text)
		debug_vn.extend(vn_text)

		# Remove empty sentences
		zh_text = [sent for sent in zh_text if sent]
		vn_text = [sent for sent in vn_text if sent]

		zh_num = len(zh_text)
		vn_num = len(vn_text)
		
		# Create index for each sentence
		index_src += zh_num
		index_tgt += vn_num

		if zh_num == 0 or vn_num == 0:
			continue

		index_golden.append( (index_src - 1, index_tgt - 1) )

	if enable_debug:
		# Write the debug information to a file
		with open("debug_zh.txt", "w", encoding="utf-8") as f:
			for line in debug_zh:
				f.write(line + "\n")
				
		with open("debug_vn.txt", "w", encoding="utf-8") as f:
			for line in debug_vn:
				f.write(line + "\n")

	return index_golden