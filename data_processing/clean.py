with open('../data/dataset/v2/dialogue_data/context_2_20/test_context_text.txt') as textlines:
	for text_context in textlines:
		utterances = text_context.split('|')
		if len(utterances) == 1:
			print('dd',text_context)