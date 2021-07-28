import os
import json
import pickle as pkl
import argparse
import logging
from collections import Counter
import pandas as pd
# KB == Knowledge base
# KB_LIST = ['compulsory_fields', 'search_criteria']

class KB():
	def __init__(self, data_dir, out_dir):
		logging.basicConfig(level=logging.INFO)
		self.logger = logging.getLogger(' Knowledge base')
		self.data_dir = data_dir
		self.out_dir = out_dir
		self.criteria_counter = Counter()
		# self.criteria_val_counter = Counter()
		self.sc_file = os.path.join(self.out_dir,"search_criteria.txt")
		# self.cf_file = os.path.join(self.out_dir,"compulsory_fields.txt")
		# self.diff_file = os.path.join(self.out_dir,"diff_fields.txt")
		# self.name_file = os.path.join(self.out_dir,"file_name.txt")
		self.sc_pkl = os.path.join(self.out_dir,"search_criteria.pkl")
		# self.sc_val_pkl = os.path.join(self.out_dir,"search_criteria_val.pkl")

	def save_to_pickle(self, obj, filename):
		if os.path.isfile(filename):
			self.logger.info(" Overwriting %s." % filename)
		else:
			self.logger.info(" Saving to %s." % filename)
		with open(filename, 'wb') as f:
			pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)

	# def split_text_column(self, df, colname, delimiter):
	# 	data = pkl.load(open('search_criteria_val.pkl','r'))
	# 	df = pd.DataFrame.from_dict(data, orient='index').reset_index()
	# 	df.columns = ['kb','count']
	# 	temp_df = df['kb'].apply(lambda x: x.split('|||'))
	# 	df3 = pd.DataFrame(temp_df.values.tolist(), columns=['a','b','c'])
	# 	temp_df = df[colname].str.split(delimiter, expand=True)
	# 	temp_df.columns = ['criteria','sub_criteria','val']
	# 	# temp_df['val'] = pd.to_numeric(temp_df['val'])
	# 	df = df.join(temp_df).drop(colname, axis=1)
	# 	return df

	def create_kb(self):
		self.read_jsondir(self.data_dir)
		self.save_to_pickle(self.criteria_counter, self.sc_pkl)
		# self.save_to_pickle(self.criteria_val_counter, self.sc_val_pkl)

	def read_jsondir(self, json_dir):
		for file in os.listdir(json_dir):
			if file.endswith('.json'):
				self.read_jsonfile(os.path.join(json_dir, file))

	def join_and_append(self, local_list, global_list):
		line = ','.join(local_list)
		global_list.append(line)

	def write_list_to_file(self, out_file_path, out_list):
		with open(out_file_path, 'a+') as out_file:
		    for item in out_list:
		        out_file.write("{}\n".format(item))

	def read_jsonfile(self, json_file):
		try:
			dialogue = json.load(open(json_file))
		except:
			print(json_file)
			return None
		filter(None, dialogue)
		sc_list=[]
		cf_list=[]
		file_list=[]
		diff_list=[]
		for utterance in dialogue:
			if 'kb' in utterance['utterance']:
				kb_line = utterance['utterance']
				sc = kb_line['kb']
				sc_keys = sc
				self.criteria_counter.update(sc_keys)
				self.join_and_append(sc_keys, sc_list)
		self.write_list_to_file(self.sc_file, sc_list)


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-data_dir', type=str, help='dir path')
	parser.add_argument('-out_dir', type=str, help='dir path')
	parser.add_argument('-data_type', type=str, help='dir path')
	args = parser.parse_args()
	data_dir = os.path.join(args.data_dir, args.data_type)
	kb = KB(data_dir, args.out_dir)
	kb.create_kb()

