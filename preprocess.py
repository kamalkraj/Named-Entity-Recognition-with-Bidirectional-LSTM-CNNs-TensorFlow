import re
from collections import defaultdict

import numpy as np


def iob_to_iob2(tags):
	prev = "O"
	for i in range(len(tags)):
		tag = re.sub(r'^B-|^I-', '',  tags[i])
		if tags[i].startswith("I-") and not prev.endswith("-"+tag):
			tags[i] = "B-"+tag
		prev = tags[i]
	return tags


def read_data(file):
	''' reading data '''
	words = []
	tags = []

	output = []

	with open(file, 'r') as data:
		for line in data:
			if line.strip():
				vals = line.strip().split(" ")
				if vals[0] != "-DOCSTART-":
					words.append(vals[0])
					tags.append(vals[-1])
			elif len(words) > 0:
				tags = iob_to_iob2(tags)
				output.append([(word, tag) for word, tag in zip(words, tags)])
				words = []
				tags = []
	return output

def lst_to_array(words, tags, max_sent_len):
	output = np.zeros((1, max_sent_len * 2 + 1), np.int32)
	output[0, :len(words)] = words
	output[0, max_sent_len:max_sent_len + len(tags)] = tags
	output[0, -1] = len(words)
	return output

def prepare_sent(vals, word_to_id, tag_to_id, max_sent_len):
	words = [word_to_id[word[0]] for word in vals]
	tags = [word[-1] for word in vals]
	tags_lst = []

	for i in range(len(tags)):

		if tags[i] == "O":
			tags_lst.append(0)
			continue

		tag = re.sub(r'^B-|^I-', '',  tags[i])
		if i != len(tags) - 1 and tags[i].startswith('B-') and not tags[i + 1].startswith('I-'):
			tags_lst.append(tag_to_id['U-' + tag])
		elif i != len(tags) - 1 and tags[i].startswith('B-') and tags[i + 1].startswith('I-'):
			tags_lst.append(tag_to_id['B-' + tag])
		elif i != len(tags) - 1 and tags[i].startswith('I-') and tags[i + 1].startswith('I-'):
			tags_lst.append(tag_to_id['I-' + tag])
		elif i != len(tags) - 1 and tags[i].startswith('I-') and not tags[i + 1].startswith('I-'):
			tags_lst.append(tag_to_id['L-' + tag])

		# last index
		elif i == len(tags) - 1 and tags[i].startswith('I-'):
			tags_lst.append(tag_to_id['L-' + tag])
		elif i == len(tags) - 1 and tags[i].startswith('B-'):
			tags_lst.append(tag_to_id['U-' + tag])

	tags = tags_lst

	return lst_to_array(words, tags, max_sent_len)

if __name__ == "__main__":
	
	MAX_SENT_LEN = 124

	train = read_data("data/train.txt")
	test = read_data("data/test.txt")
	valid = read_data("data/valid.txt")

	word2idx = {}

	for dataset in [train, test, valid]:
		for sentence in dataset:
			for token, label in sentence:
				word2idx[token] = len(word2idx)

	label2idx = {'B-LOC': 4, 'B-MISC': 2, 'B-ORG': 3, 'B-PER': 1, 'I-LOC': 8, 'I-MISC': 6, 'I-ORG': 7, 'I-PER': 5,
				 'L-LOC': 12, 'L-MISC': 10, 'L-ORG': 11, 'L-PER': 9, 'O': 0, 'U-LOC': 16, 'U-MISC': 14, 'U-ORG': 15, 'U-PER': 13}
	
	def prepare(dataset):
		for i,sentence in enumerate(dataset):
			dataset[i] = prepare_sent(sentence, word2idx, label2idx, MAX_SENT_LEN)
		return dataset
	
	np.save("processed_data/train",prepare(train))
	np.save("processed_data/test",prepare(test))
	np.save("processed_data/valid",prepare(valid))
	np.save("processed_data/word2idx",word2idx)
	np.save("processed_data/tag2idx",label2idx)