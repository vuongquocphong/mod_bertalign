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
		return sents
	
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