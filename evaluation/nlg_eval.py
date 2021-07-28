import sys
sys.path.append('/home/sagarwal/projects/mmd/nlg-eval/')
from nltk.translate.bleu_score import corpus_bleu
# from nlgeval import compute_metrics
import argparse

def main(args):
	# metrics_dict = compute_metrics(hypothesis=args.pred_file, 
	# 				references=[args.ref_file], 
	# 				no_skipthoughts=True, no_glove=True)

	pred_file = open(args.pred_file,'r')
	pred = pred_file.readlines()
	ref_file = open(args.ref_file,'r')
	ref = ref_file.readlines()

	references = [[ref[i].replace('\n','').split(' ')] for i in range(len(ref))]
	candidates = [pred[i].replace('\n','').split(' ') for i in range(len(pred))]

	# print(references, candidates)
	score = corpus_bleu(references, candidates)
	print(score)

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-pred_file', type=str, help='hypothesis file')
	parser.add_argument('-ref_file', type=str, help='reference file')
	args = parser.parse_args()
	main(args)
