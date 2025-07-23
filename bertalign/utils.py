import re
import requests
from googletrans import Translator
from collections import defaultdict
from underthesea import sent_tokenize

def clean_text(text, lang):
	clean_text = []
	text = text.strip()
	lines = text.splitlines()
	for line in lines:
		line = line.strip()
		if line:
			line = re.sub('\s+', '', line) if lang == 'zh' else re.sub(r'\s+', ' ', line)
			clean_text.append(line)
	return "\n".join(clean_text)

def detect_lang(text):
    translator = Translator(service_urls=[
      'translate.google.com.hk',
    ])
    max_len = 200
    chunk = text[0 : min(max_len, len(text))]
    lang = translator.detect(chunk).lang
    if lang.startswith('zh'): lang = 'zh'
    return lang

def length_vi(text):

	text = re.sub(r'\s+', ' ', text)
	length = len(text)

	text = re.sub(r'[^\w\s]', '', text)

	length -= len(text)
	text = text.lower()
	
	characters = text.split(' ')
	length += len(characters)

	return length
	
def split_sents(text, lang):
	if lang == 'zh':
		sents = _split_zh(text)
		return sents
	
	sents = sent_tokenize(text)
	sents = [sent.strip() for sent in sents]

	refine_sents = [sents[-1]] 
	index = len(sents) - 2
	while index >= 0:
		if not re.match(r'^.*?:?\s*\d+\s*\.$', sents[index]):
			refine_sents.append(sents[index])
			index -= 1
			continue

		if not re.match(r'^\d+\s*\.$', sents[index]):
			refine_sents.append(sents[index])
			index -= 1
			continue

		refine_sents[-1] = sents[index] + ' ' + refine_sents[-1]
		index -= 1
	
	refine_sents.reverse()
	return refine_sents
	
def _split_zh(text, limit=1000):
	sent_list = []

	text = re.sub('(?P<quotation_mark>([。.？?！!](?![”’"」\'）])))', r'\g<quotation_mark>\n', text)
	text = re.sub('(?P<quotation_mark>([。.？?！!]|…{1,2})[”’"」\'）])', r'\g<quotation_mark>\n', text)
	
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
# OVERLAP PREPARATION
###########################################################################
TRANSLITERATE_URL = "https://tools.clc.hcmus.edu.vn/api/web/clc-sinonom/sinonom-transliteration"

def _post_request_to_api( lines: list[str], is_split: bool = False ) -> list[str]:
	"""
	Sends a POST request to the specified API.

	:param data: The text need to convert to sino-vietnamese.
	:return: The list of sino-converted of each sentence.
	"""

	def send_single_api_request(text: str, url=TRANSLITERATE_URL) -> str:
		payload = {
			"text": text
		}
		
		headers = {
			"User-Agent": "Mozilla/5.0",
			"Referer": "https://tools.clc.hcmus.edu.vn",
			"Origin": "https://tools.clc.hcmus.edu.vn",
			"Content-Type": "application/json"
		}

		try:
			response = requests.post(url, json=payload, headers=headers)
			response.raise_for_status()  # Raise error for bad status codes
			data = response.json()

			if data.get("is_success") and "data" in data:
				result = data["data"].get("result_text_transcription")
				if result:
					to_return = " ".join(result) if isinstance(result, list) else result
					return to_return
				else: raise ValueError("No transcription found in response.")
			else: raise ValueError("API returned failure or malformed response.")

		except Exception as e: raise RuntimeError(f"API request failed: {e}")
     
	def batch_transliterate(sentences_generator):
		"""
		Process sentences from a generator for memory efficiency.
		
		:param sentences: Generator or iterable of sentences
		:param server_url: API server URL
		:return: List of transliterated sentences
		"""
		results = []

		for sentence in sentences_generator:
			if len(sentence) <= 100:
				try:
					response = send_single_api_request(sentence)
					results.append(response)
				except Exception as e:
					results.append(f"[Error: {e}]")
			
			else:
				chunks = re.split(r'[，,；;]', sentence)
				translated_chunks = []
				for chunk in chunks:
					try:
						response = send_single_api_request(chunk)
						translated_chunks.append(response)
					except Exception as e:
						translated_chunks.append(f"[Error: {e}]")
				
				translated_chunks = ' '.join( translated_chunks )
				results.append( translated_chunks )
		return results
	
	transliterated_lines = batch_transliterate(lines)
	return transliterated_lines
	
def _clean_zh_text(text: str) -> str:
	"""
	Cleans the input text by removing unwanted characters.

	:param text: The input text to clean.
	:return: The cleaned text.
	"""
	# Define a regex pattern to remove unwanted characters
	pattern = r"[。！？；：，—、“”‘’《》【】（）；,;:.!?\[\]\"'、]"
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

def convert_zh( src_sents: list[str], overlaps: int, is_split: bool = False):
	"""
	Convert the input text to sino-vietnamese using the API.

	:param text: The input text to convert.
	:return: A list of sino-converted sentences.
	"""

	converted_sentences = _post_request_to_api(src_sents, is_split)
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