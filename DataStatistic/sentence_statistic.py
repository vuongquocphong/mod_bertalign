import re

import pandas as pd
from underthesea import sent_tokenize


def clean_text(text):
	clean_text = []
	text = text.strip()
	lines = text.splitlines()
	for line in lines:
		line = line.strip()
		if line:
			line = re.sub('\s+', ' ', line)
			clean_text.append(line)
	return "\n".join(clean_text)
	
def split_sents(text, lang):
	if lang == 'zh':
		sents = split_zh(text)
		return sents
	
	sents = sent_tokenize(text)
	sents = [sent.strip() for sent in sents]

	refine_sents = [sents[-1]] 
	index = len(sents) - 2
	while index >= 0:
		
		if re.match(r'^.*?:\s*\d+\s*\.$', sents[index]) or re.match(r'^\s*\d+\s*\.$', sents[index]):
			refine_sents[-1] = sents[index] + ' ' + refine_sents[-1]
		
		else: refine_sents.append(sents[index])
		index -= 1
	
	refine_sents.reverse()
	return refine_sents
	
def split_zh(text, limit=1000):

	# Proposed version
	text = re.sub('(?P<quotation_mark>([。.？?！!](?![”’"」\'）])))', r'\g<quotation_mark>\n', text)
	text = re.sub('(?P<quotation_mark>([。.？?！!]|…{1,2})[”’"」\'）])', r'\g<quotation_mark>\n', text)
	
	sent_list = []
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

def resolve_folder( folder_path ):
	
	src_path = folder_path + "/chinese_pars.txt"
	tgt_path = folder_path + "/translation_pars.txt"
	golden_path = folder_path + "/golden.txt"

	if folder_path == "/home/hoktro/mod_bertalign/Data/dai_viet_su_ki":
		src_path = folder_path + "/train_src_pars.txt"
		tgt_path = folder_path + "/train_tgt_pars.txt"
		golden_path = folder_path + "/train_gold.txt"
	
	src_sents, tgt_sents = [], []

	# Get source text
	with open(src_path, "r", encoding = "utf8") as f:
		src_paragraphs = f.readlines()
	
	for paragraph in src_paragraphs:
		paragraph = clean_text( paragraph )
		sents = split_sents(paragraph, "zh")
		src_sents.extend(sents)
	
	# Get target text
	with open(tgt_path, "r", encoding = "utf8") as f:
		tgt_paragraphs = f.readlines()
	
	for paragraph in tgt_paragraphs:
		paragraph = clean_text( paragraph )
		sents = split_sents(paragraph, "vi")
		tgt_sents.extend(sents)

	# Get golden text
	with open(golden_path, "r", encoding = "utf8") as f:
		golden_lines = f.readlines()
	
	src_limit, tgt_limit = len(src_sents), len(tgt_sents)
	src_current, tgt_current = 0, 0
	alignments_type = []

	for i, line in enumerate(golden_lines):
		line = line.split("\t")

		first_part, second_part = line
		first_part = first_part.strip()
		second_part = second_part.strip()

		_, __ = 0, 0

		for src_index in range(src_current, src_limit):
			src_bead = src_sents[src_current : src_index + 1]
			length_src = sum(len(sent) for sent in src_bead)

			if length_src > len(first_part):
				_ = src_index - src_current
				break
		
		for tgt_index in range(tgt_current, tgt_limit):
			tgt_bead = tgt_sents[tgt_current : tgt_index + 1]
			length_tgt = sum(len(sent) for sent in tgt_bead) + ( tgt_index - tgt_current - 1 )

			if length_tgt > len(second_part):
				__ = tgt_index - tgt_current
				break
		
		if i == len(golden_lines) - 1:
			_ = src_limit - src_current
			__ = tgt_limit - tgt_current

		src_bead = src_sents[src_current : src_current + _]
		tgt_bead = tgt_sents[tgt_current : tgt_current + __]

		src_bead = "".join(src_bead)
		tgt_bead = " ".join(tgt_bead)

		if len(first_part) - len(src_bead) > 5 or len(second_part) - len(tgt_bead) > 5:
			print(f"Mismatch detected: alignments type: {(_, __)}")
			print(f"Source bead: {src_bead}")
			print(f"First part: {first_part}")
			print(f"Target bead: {tgt_bead}")
			print(f"Second part: {second_part}")
			print("======================================")
			break

		
		alignments_type.append((first_part, second_part, _, __))

		src_current = src_current + _
		tgt_current = tgt_current + __

		global_count.append((_, __))

	# # Export alignments type to a file
	# with open("/home/hoktro/mod_bertalign/DataStatistic/alignments_type.txt", "w", encoding="utf8") as f:
	# 	for alignment in alignments_type:
	# 		f.write(f"{alignment[0]}\t{alignment[1]}\t{alignment[2]}\t{alignment[3]}\n")
	
	# # Export splitted sentences to files
	# with open("/home/hoktro/mod_bertalign/DataStatistic/splitted_src.txt", "w", encoding="utf8") as f:
	# 	for sent in src_sents:
	# 		f.write(f"{sent}\n")
	
	# with open("/home/hoktro/mod_bertalign/DataStatistic/splitted_tgt.txt", "w", encoding="utf8") as f:
	# 	for sent in tgt_sents:
	# 		f.write(f"{sent}\n")

global_count = []

if __name__ == "__main__":
	folder_path = "/home/hoktro/mod_bertalign/Data/dai_nam_chinh_bien_liet_truyen"
	resolve_folder(folder_path)
	folder_path = "/home/hoktro/mod_bertalign/Data/tam_quoc_dien_nghia/tqdn1"
	resolve_folder(folder_path)
	folder_path = "/home/hoktro/mod_bertalign/Data/dai_viet_su_ki"
	resolve_folder(folder_path)

	# Couunt each type of alignment
	counts = {}
	for src_count, tgt_count in global_count:
		if (src_count, tgt_count) not in counts:
			counts[(src_count, tgt_count)] = 0
		counts[(src_count, tgt_count)] += 1

	# Sort the count by key
	counts = dict(sorted(counts.items(), key=lambda item: item[0]))

	# Export the count to a xlsx file
	counts_list = [(src_count, tgt_count, count) for (src_count, tgt_count), count in counts.items()]
	df = pd.DataFrame(counts_list, columns=["Source Count", "Target Count", "Count"])
	df.to_excel("/home/hoktro/mod_bertalign/DataStatistic/alignments_count.xlsx", index=False)
	print("Alignment counts exported to alignments_count.xlsx")
	print("Total alignments:", len(global_count))
	print("Alignment types:", len(counts))