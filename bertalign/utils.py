from collections import defaultdict
from itertools import chain
import re
import xmlrpc.client
from sentence_splitter import SentenceSplitter
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
		sents = _split_zh(text)
	else:
		sents = sent_tokenize(text)
		sents = [sent.strip() for sent in sents]
	return sents
	
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
		
def yield_overlaps(lines, num_overlaps):
	lines = [_preprocess_line(line) for line in lines]
	for overlap in range(1, num_overlaps + 1):
		for out_line in _layer(lines, overlap):
			# check must be here so all outputs are unique
			out_line2 = out_line[:10000]  # limit line so dont encode arbitrarily long sentences
			yield out_line2

def _layer(lines, num_overlaps, comb=' '):
	if num_overlaps < 1:
		raise Exception('num_overlaps must be >= 1')
	out = ['PAD', ] * min(num_overlaps - 1, len(lines))
	for ii in range(len(lines) - num_overlaps + 1):
		out.append(comb.join(lines[ii:ii + num_overlaps]))
	return out
	
def _preprocess_line(line):
	line = line.strip()
	if len(line) == 0:
		line = 'BLANK_LINE'
	return line

###########################################################################
# UNION PREPARATION
###########################################################################

def load_nom_dict(file_path: str) -> dict[str, str]:
	"""
	Load the nom dictionary from a file.

	:param file_path: The path to the dictionary file.
	:return: A dictionary with characters as keys and their replacements as values.
	"""
	nom_dict = {}
	# load the excel file
	import pandas as pd
	df = pd.read_excel(file_path)
 
	# extract the first and the second columns
	chinese = df.iloc[:, 0].tolist()
	vietnamese = df.iloc[:, 1].tolist()
	# create a dictionary from the two columns
 
	if len(chinese) != len(vietnamese):
		raise ValueError("The two columns must have the same length.")
	for i in range(len(chinese)):
		if chinese[i] in nom_dict:
			continue
		nom_dict[chinese[i]] = vietnamese[i]
	
	return nom_dict

nom_dict = load_nom_dict('bertalign/dictionary/D_203_single_char_nom_qn_dictionary_thi_vien.xlsx')

def _post_request_to_api( data: str ) -> list[str]:
	"""
	Sends a POST request to the specified API.

	:param data: The text need to convert to sino-vietnamese.
	:return: The list of sino-converted of each sentence.
	"""
	def batch_transliterate(sentences, server_url="http://localhost:8080/RPC2"):
		server = xmlrpc.client.ServerProxy(server_url)
		results = []

		for sentence in sentences:
			params = {'text': sentence}
			try:
				response = server.translate(params)
				results.append(response.get('text', ''))  # fallback to empty string if 'text' missing
			except Exception as e:
				results.append(f"[Error: {e}]")  # include error for debugging
		return results
	
	def preprocess_snt_for_transliteration(replace_dict, text: str):
		# Preprocess the text if necessary (e.g., remove unwanted characters)
		for char in text:
			# Check if the character is in Basic Multilingual Plane (BMP)
			if ord(char) <= 65535:
				continue
			# If not, check if it is in the replace dictionary
			if char in replace_dict.keys():
				text = text.replace(char, replace_dict[char])
				print(replace_dict[char])
			else:
				# If the character is not in the replace dictionary, replace it with a space
				text = text.replace(char, ' ')
		return text
	
	lines = _split_zh(data, limit=1000)
 
	spaced_lines = []
	
	for line in lines:
		line = ' '.join(line)
		spaced_lines.append(line)

	preprocessed_lines = [preprocess_snt_for_transliteration(nom_dict, line) for line in spaced_lines]  

	transliterated_lines = batch_transliterate(preprocessed_lines)

	return transliterated_lines
	
def _clean_zh_text(text: str) -> str:
	"""
	Cleans the input text by removing unwanted characters.

	:param text: The input text to clean.
	:return: The cleaned text.
	"""
	# Define a regex pattern to remove unwanted characters
	pattern = r"[。！？；：，—“”‘’《》【】（）；,;:.!?]"
	return re.sub(pattern, '', text)

def _clean_vietnamese_text(text: str) -> str:
	"""
	Cleans the Vietnamese text by removing unwanted characters.

	:param text: The input text to clean.
	:return: The cleaned text.
	"""
	# Define a regex pattern to match Vietnamese sentence marks
	pattern = r"[.!?；：，—“”‘’\[\]\(\),:;\"]"
	return re.sub(pattern, ' ', text)

def convert_zh(text: str, overlaps: int):
	"""
	Convert the input text to sino-vietnamese using the API.

	:param text: The input text to convert.
	:return: A list of sino-converted sentences.
	"""

	converted_sentences = _post_request_to_api(text)
	if converted_sentences is None:
		raise Exception("Error in API response.")
	
	# Initiate the words list
	words = [[] for _ in range(overlaps)]

	# First layer
	words[0].extend(
		[
			[word.strip() for word in _clean_zh_text(sentence).split() if word.strip()]
			for sentence in converted_sentences
		]
	)

	num_sent = len(words[0])

	# Remaining layers	  
	for layer in range(2, overlaps + 1):
		index = layer - 1
		for sent in words[layer - 2]:
			if index >= num_sent: break
			words[layer - 1].append(sent + words[0][index])
			index += 1
	
	# Add PAD for all layers
	for layer in range(2, overlaps + 1):
		words[layer - 1] = [['PAD']] * min( layer - 1, num_sent) + words[layer - 1]

	# Create words length list
	src_words_len = [[] for _ in range(overlaps)]
	for layer in range(overlaps):
		src_words_len[layer] = [len(sent) for sent in words[layer]]
	
	return words, src_words_len

def convert_vn(tgt: list[str], overlaps: int) -> list[list[str]]:
	"""
	Convert the input text to sino-vietnamese using the API.

	:param text: The input text to convert.
	:return: A list of sino-converted sentences.
	"""
	# Initiate the words list
	result = [[] for _ in range(overlaps)]

	# First layer
	result[0].extend(
		[
			[word.strip() for word in _clean_vietnamese_text(sentence).split() if word.strip()]
			for sentence in tgt
		]
	)

	num_sent = len(result[0])

	# Remaining layers	  
	for layer in range(2, overlaps + 1):
		index = layer - 1
		for sent in result[layer - 2]:
			if index >= num_sent: break
			result[layer - 1].append(sent + result[0][index])
			index += 1
	
	# Add PAD for all layers
	for layer in range(2, overlaps + 1):
		result[layer - 1] = [['PAD']] * min( layer - 1, num_sent) + result[layer - 1]

	# Create words length list
	tgt_words_len = [[] for _ in range(overlaps)]
	for layer in range(overlaps):
		tgt_words_len[layer] = [len(sent) for sent in result[layer]]
	
	return result, tgt_words_len

def _create_dict_from_list(lst: list[str]) -> dict[str, list[int]]:
	"""
	Creates a dictionary from a list of words.

	:param lst: The list of words.
	:return: A dictionary with words as keys and their indices as values.
	"""
	reserve = defaultdict(list[int])
	for index, word in enumerate(lst):
		reserve[word].append(index)

	for word in reserve:
		reserve[word].reverse()

	return reserve

def convert_words_to_indexList(words: list[list[str]], overlaps: int) -> list[dict[str, list[int]]]:
	"""
	Convert the words to index list.

	:param words: The list of words.
	:return: A list of dictionaries with words as keys and their indices as values.
	"""
	result = [[] for _ in range(overlaps)]
	for layer in range(overlaps):
		result[layer] = [_create_dict_from_list(sent) for sent in words[layer]]
	return result