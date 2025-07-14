import os
import re
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
		if not re.match(r'^\d+\s*\.$', sents[index]):
			refine_sents.append(sents[index])
			index -= 1
			continue

		refine_sents[-1] = sents[index] + ' ' + refine_sents[-1]
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

def clean_zh_text(text):
	
	# Remove all spaces
	text = re.sub(r'\s+', '', text)

	# Remove all except for Chinese characters
	text = re.sub(r'[^\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf\uf900-\ufaff]', '', text)
	
	# Split characters
	characters = list(text)

	return characters

def clean_vi_text(text):
	
	# Remove all except for Vietnamese characters and some punctuation
	text = re.sub(r'[^\w\s]', '', text)

	# Remove all spaces
	text = re.sub(r'\s+', ' ', text)

	text = text.lower()
	
	# Split characters
	characters = text.split(' ')

	return characters

global_dict_src = set()
global_dict_tgt = set()

def process_data(folder_path):

	global global_dict_src, global_dict_tgt

	src_path = folder_path + '/chinese_pars.txt'
	tgt_path = folder_path + '/translation_pars.txt'
	golden_path = folder_path + '/alignments.txt'

	total_src, totalLen_src, totalChar_src = 0, 0, 0
	total_tgt, totalLen_tgt, totalChar_tgt = 0, 0, 0

	local_dict_src = set()
	local_dict_tgt = set()

	# Resolve source text
	with open(src_path, 'r', encoding='utf-8') as f:
		src_text = f.readlines()
	for src_par in src_text:
		sentences = split_sents(clean_text(src_par), 'zh')

		total_src += len(sentences)
		
		for sent in sentences:
			totalLen_src += len(sent)
			
			chars = clean_zh_text(sent)
			totalChar_src += len(chars)

			local_dict_src.update(chars)
			global_dict_src.update(chars)
	
	averaveLen_src = totalLen_src / total_src if total_src > 0 else 0

	# Resolve target text
	with open(tgt_path, 'r', encoding='utf-8') as f:
		tgt_text = f.readlines()
	
	for tgt_par in tgt_text:
		sentences = split_sents(clean_text(tgt_par), 'vi')

		total_tgt += len(sentences)
		
		for sent in sentences:
			totalLen_tgt += len(sent)
			
			chars = clean_vi_text(sent)
			totalChar_tgt += len(chars)

			local_dict_tgt.update(chars)
			global_dict_tgt.update(chars)
	
	averaveLen_tgt = totalLen_tgt / total_tgt if total_tgt > 0 else 0

	# Resolve golden text
	with open(golden_path, 'r', encoding='utf-8') as f:
		golden_text = f.readlines()

	golden_bead, golden_proportion = 0, 0.00
	
	for line in golden_text:
		txt = line.split('\t')
		if len(txt) < 2:
			raise ValueError("Golden text format error: {}".format(line))
		
		left, right = txt[0].strip(), txt[1].strip()

		if len(left) == 0 or len(right) == 0:
			continue

		golden_bead += 1
		golden_proportion += len(left) / len(right)

	average_proportion = golden_proportion / golden_bead if golden_bead > 0 else 0

	return total_src, total_tgt, round(averaveLen_src, 2), round(averaveLen_tgt, 2), totalChar_src, len(local_dict_src), totalChar_tgt, len(local_dict_tgt), round(average_proportion, 4)

def process_data_SKTMT(folder_path):

	global global_dict_src, global_dict_tgt

	total_src, totalLen_src, totalChar_src = 0, 0, 0
	total_tgt, totalLen_tgt, totalChar_tgt = 0, 0, 0

	golden_bead, golden_proportion = 0, 0.00

	local_dict_src = set()
	local_dict_tgt = set()

	# Get all file end with .par
	files = [f for f in os.listdir(folder_path + "/zh") if f.endswith('.par')]

	for file in files:
		src_file_path = os.path.join(folder_path, "zh", file)
		tgt_file_path = os.path.join(folder_path, "vi", file)

		golden_name = file + '_alignments.txt'
		golden_path = os.path.join(folder_path, golden_name)

		if not os.path.exists(src_file_path) or not os.path.exists(tgt_file_path):
			print(f"File not found: {src_file_path} or {tgt_file_path}")
			continue

		# Resolve source text
		with open(src_file_path, 'r', encoding='utf-8') as f:
			src_text = f.readlines()
		for src_par in src_text:
			sentences = split_sents(clean_text(src_par), 'zh')

			total_src += len(sentences)
			
			for sent in sentences:
				totalLen_src += len(sent)
				
				chars = clean_zh_text(sent)
				totalChar_src += len(chars)

				local_dict_src.update(chars)
				global_dict_src.update(chars)

		# Resolve target text
		with open(tgt_file_path, 'r', encoding='utf-8') as f:
			tgt_text = f.readlines()
		
		for tgt_par in tgt_text:
			sentences = split_sents(clean_text(tgt_par), 'vi')

			total_tgt += len(sentences)
			
			for sent in sentences:
				totalLen_tgt += len(sent)
				
				chars = clean_vi_text(sent)
				totalChar_tgt += len(chars)

				local_dict_tgt.update(chars)
				global_dict_tgt.update(chars)

		# Resolve golden text
		if not os.path.exists(golden_path):
			print(f"Golden file not found: {golden_path}")
			continue
		
		with open(golden_path, 'r', encoding='utf-8') as f:
			golden_text = f.readlines()

		for line in golden_text:
			txt = line.split('\t')
			if len(txt) < 2:
				raise ValueError("Golden text format error: {}".format(line))
			
			left, right = txt[0].strip(), txt[1].strip()

			if len(left) == 0 or len(right) == 0:
				continue

			golden_bead += 1
			golden_proportion += len(left) / len(right)

	averaveLen_src = totalLen_src / total_src if total_src > 0 else 0
	averaveLen_tgt = totalLen_tgt / total_tgt if total_tgt > 0 else 0
	average_proportion = golden_proportion / golden_bead if golden_bead > 0 else 0

	return total_src, total_tgt, round(averaveLen_src, 2), round(averaveLen_tgt, 2), totalChar_src, len(local_dict_src), totalChar_tgt, len(local_dict_tgt), round(average_proportion, 4)

def analyze_data():

	results = []

	# Data model
	prefix = "/home/hoktro/mod_bertalign/Data/"
	folder_paths = [
		"dai_nam_chinh_bien_liet_truyen",
		"DVSK",
		"TQDN/TQDN_01",
		"TQDN/TQDN_02",	
		"TQDN/TQDN_03",
	]

	for folder in folder_paths:
		folder_path = prefix + folder
		print("Processing folder: {}".format(folder_path))
		
		try:
			result = process_data(folder_path)
			results.append((folder, *result))
		except Exception as e:
			print("Error processing folder {}: {}".format(folder_path, e))

	# Data Lab
	prefix = "/home/hoktro/mod_bertalign/DataLab/"
	folder_paths = [
		"TongSuViet",
		"TrietGia",
		"TrietHoc/PBS_001",
		"TrietHoc/PKS_001",
		"TrietHoc/PVS_003",
		"TuThu",
		"VanBia",	
	]

	for folder in folder_paths:
		folder_path = prefix + folder
		print("Processing folder: {}".format(folder_path))
		
		try:
			result = process_data(folder_path)
			results.append((folder, *result))
		except Exception as e:
			print("Error processing folder {}: {}".format(folder_path, e))

	# Data SKTMT
	folder_path = "/home/hoktro/mod_bertalign/sktmt"
	print("Processing SKTMT folder: {}".format(folder_path))
	try:
		result = process_data_SKTMT(folder_path)
		results.append(("SKTMT", *result))
	except Exception as e:
		print("Error processing SKTMT folder: {}".format(e))

	# Total statistics
	total_src = sum(res[1] for res in results)
	total_tgt = sum(res[2] for res in results)
	totalLen_src = sum(res[3] * res[1] for res in results)
	totalLen_tgt = sum(res[4] * res[2] for res in results)
	totalChar_src = sum(res[5] for res in results)
	totalChar_tgt = sum(res[6] for res in results)
	uniqueChar_src = len(global_dict_src)
	uniqueChar_tgt = len(global_dict_tgt)
	average_proportion = sum(res[9] for res in results) / len(results)

	results.append(("Total", total_src, total_tgt,
		round(totalLen_src / total_src, 2) if total_src > 0 else 0,
		round(totalLen_tgt / total_tgt, 2) if total_tgt > 0 else 0,
		totalChar_src, uniqueChar_src, totalChar_tgt, uniqueChar_tgt,
		round(average_proportion, 4)))

	# Write into xlsx file
	import pandas as pd
	df = pd.DataFrame(results, columns=[
		'Folder', 'Total Source', 'Total Target', 
		'Average Length Source', 'Average Length Target', 
		'Total Characters Source', 'Unique Characters Source', 
		'Total Characters Target', 'Unique Characters Target', 
		'Average Proportion'
	])
	output_path = "/home/hoktro/mod_bertalign/DataStatistic/statistic_results.xlsx"
	df.to_excel(output_path, index=False)
	print("Results saved to {}".format(output_path))

if __name__ == "__main__":
	analyze_data()