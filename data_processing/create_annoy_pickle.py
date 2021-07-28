import pickle as pkl 
my_dict_final = {}
# with open('../data/dataset/v2/dialogue_data/context_2_20/train_annoy.pkl', 'rb') as f:
# 	data_1 = (pkl.load(f))

# print(len(data_1.keys()))

with open('../data/dataset/v2/dialogue_data/context_2_20/train_annoy.pkl', 'rb') as f:
	my_dict_final.update(pkl.load(f))

with open('../data/dataset/v2/dialogue_data/context_2_20/valid_annoy.pkl', 'rb') as f:
	my_dict_final.update(pkl.load(f))

with open('../data/dataset/v2/dialogue_data/context_2_20/test_annoy.pkl', 'rb') as f:
	my_dict_final.update(pkl.load(f))

# print()
# filename = '../data/raw_catlog/image_annoy_index/ImageUrlToIndex.pkl'
filename1 = '../../rizwan/mmd/data/raw_catlog/image_annoy_index/ImageUrlToIndex.pkl'

with open(filename1, 'wb') as f:
	pkl.dump(my_dict_final, f, protocol=pkl.HIGHEST_PROTOCOL)

print('pp',len(my_dict_final))



# # print(len(data_1))
# i = 0
# # for key in my_dict_final:
# # 	print(key, my_dict_final[key])
# # 	i += 1
# # print(i)

# print(len(my_dict_final.keys()))

# filename1 = '../../rizwan/mmd/data/raw_catlog/image_annoy_index/ImageUrlToIndex.pkl'

	

# with open(filename1, 'rb') as f:
# 	data_1 = (pkl.load(f))

# print('dd',len(data_1.keys()))
