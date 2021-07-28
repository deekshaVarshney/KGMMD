import pandas as pd

'''
hred=[]
mhred=[]
mhred_attn=[]
hred_kb=[]
mhred_kb=[]
mhred_kb_attn=[]

test_target=[]
test_context=[]

f1=open("/home1/deeksha/mmd_final/models/context_2_20/HRED/_v4/test_tokenized.txt", 'r')
n_lines=100
for i in range(n_lines):
	test_target.append(f1.readline().strip())
# print(test_target[17])

f2=open("/home1/deeksha/mmd_final/models/context_2_20/HRED/_v4/test_context_text.txt", 'r')
for i in range(n_lines):
	test_context.append(f2.readline().strip())

f3=open("/home1/deeksha/mmd_final/models/context_2_20/HRED/_v4/pred_30.txt", 'r')
for i in range(n_lines):
	hred.append(f3.readline().strip())

f4=open("/home1/deeksha/mmd_final/models/context_2_20/MultimodalHRED/_v4/pred_30.txt", 'r')
for i in range(n_lines):
	mhred.append(f4.readline().strip())

f5=open("/home1/deeksha/mmd_final/models_attn/context_2_20/MultimodalHRED/_v42/pred_30.txt", 'r')
for i in range(n_lines):
	mhred_attn.append(f5.readline().strip())

f6=open("/home1/deeksha/mmd_final/models/context_2_20/HRED/models_kb/_v4/pred_30.txt", 'r')
for i in range(n_lines):
	hred_kb.append(f6.readline().strip())

f7=open("/home1/deeksha/mmd_final/models/context_2_20/MultimodalHRED/models_kb/_v4/pred_30.txt", 'r')
for i in range(n_lines):
	mhred_kb.append(f7.readline().strip())

f8=open("/home1/deeksha/mmd_final/models_attn/context_2_20/MultimodalHRED/models_kb/_v42/pred_30.txt", 'r')
for i in range(n_lines):
	mhred_kb_attn.append(f8.readline().strip())

print(len(test_context))
print(len(test_target))
print(len(hred))
print(len(mhred))
print(len(mhred_attn))
print(len(hred_kb))
print(len(mhred_kb))
print(len(mhred_kb_attn))

# print(hred)
details={
	'SRC': test_context,
	'TRG': test_target,
	'HRED': hred,
	'MHRED': mhred,
	'MHRED(+attn)': mhred_attn,
	'HRED(+kb)': hred_kb,
	'MHRED(+kb)': mhred_kb,
	'MHRED(+kb)(+attn)': mhred_kb_attn
}

df= pd.DataFrame(details)
print(df.head())

df.to_csv('hred_human_eval.csv')
'''
trans=[]
mtrans=[]
trans_kb=[]
mtrans_kb=[]

test_target=[]
test_context=[]

n_lines=100
f1=open("/home1/deeksha/mmd_final/models/context_2_20/Transformer/_v41/test_tokenized.txt", 'r')
for i in range(n_lines):
	test_target.append(f1.readline().strip())
# print(test_target[17])

f2=open("/home1/deeksha/mmd_final/models/context_2_20/Transformer/_v41/test_context_text.txt", 'r')
for i in range(n_lines):
	test_context.append(f2.readline().strip())

f3=open("/home1/deeksha/mmd_final/models/context_2_20/Transformer/_v41/pred_30.txt", 'r')
for i in range(n_lines):
	trans.append(f3.readline().strip())

f4=open("/home1/deeksha/mmd_final/models/context_2_20/MTransformer/_v41/pred_30.txt", 'r')
for i in range(n_lines):
	mtrans.append(f4.readline().strip())

f5=open("/home1/deeksha/mmd_final/models/context_2_20/Transformer/models_kb/_v41/pred_30.txt", 'r')
for i in range(n_lines):
	trans_kb.append(f5.readline().strip())

f6=open("/home1/deeksha/mmd_final/models/context_2_20/MTransformer/models_kb/_v41/pred_30.txt", 'r')
for i in range(n_lines):
	mtrans_kb.append(f6.readline().strip())

print(len(test_context))
print(len(test_target))
print(len(trans))
print(len(mtrans_kb))
print(len(mtrans))
print(len(trans_kb))

details={
	'SRC': test_context,
	'TRG': test_target,
	'Transformer': trans,
	'MTransformer': mtrans,
	'Transformer(+kb)': trans_kb,
	'MTransformer(+kb)': mtrans_kb,
}

df= pd.DataFrame(details)
print(df.head())

df.to_csv('transformer_human_eval.csv')