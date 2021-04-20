import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer_tformer import encoder_layer, decoder_layer,MTrans_img_kb_encoder_layer,MTrans_img_encoder_layer,MTrans_kb_encoder_layer,\
context_encoder_layer, MTrans_aspect_encoder_layer, MTrans_kb_aspect_encoder_layer, MTrans_img_aspect_encoder_layer, \
MTrans_img_kb_aspect_encoder_layer


class encoder(nn.Module):
	def __init__(self,input_dim,hid_dim,n_layers,n_heads,pf_dim,drop_prob=0.1):
		super().__init__()

		self.tos_embed=nn.Embedding(input_dim,hid_dim)
		self.pos_embed=nn.Embedding(input_dim,hid_dim)

		self.encoder_layer=nn.ModuleList([encoder_layer(hid_dim,pf_dim,n_heads,drop_prob) for i in range(n_layers)])
		self.dropout=nn.Dropout(p=drop_prob)
		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()

	def forward(self,src,src_mask):
		##src>>[batch_size,src_len]
		##src_mask>>[batch_size,1,1,src_len]
		batch_size=src.shape[0]
		src_len=src.shape[1]
		pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).cuda()
		pos_embedding = self.pos_embed(pos) ##[batch_size,src_len,hid_dim]
		token_embedding = self.tos_embed(src) ##[batch_size,src_len,hid_dim]
		src=self.dropout(token_embedding*(self.scale)+pos_embedding)

		##src>>[batch_size,src_len,hid_dim]
		for layer in self.encoder_layer:
			src=layer(src,src_mask)

		return src


class context_encoder(nn.Module):
	def __init__(self,input_dim,hid_dim,n_layers,n_heads,pf_dim,drop_prob=0.1):
		super(context_encoder,self).__init__()

		self.tos_embed=nn.Embedding(input_dim,hid_dim)
		self.pos_embed=nn.Embedding(input_dim,hid_dim)

		self.context_encoder_layer=nn.ModuleList([context_encoder_layer(hid_dim,pf_dim,n_heads,drop_prob) for i in range(n_layers)])
		self.dropout=nn.Dropout(p=drop_prob)
		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()
		self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
		

	def forward(self,src,src_mask,history_bank, his_mask):
		##src>>[batch_size,src_len]
		##src_mask>>[batch_size,1,1,src_len]
		batch_size=src.shape[0]
		src_len=src.shape[1]
		pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).cuda()
		pos_embedding = self.pos_embed(pos) ##[batch_size,src_len,hid_dim]
		token_embedding = self.tos_embed(src) ##[batch_size,src_len,hid_dim]
		src=self.dropout(token_embedding*(self.scale)+pos_embedding)

		##src>>[batch_size,src_len,hid_dim]
		for layer in self.context_encoder_layer:
			src=layer(src,src_mask,history_bank , his_mask)
		out = self.layer_norm(src)

		return token_embedding,out.contiguous(),src_mask



class decoder(nn.Module):
	def __init__(self,output_dim,hid_dim,n_layers,pf_dim,n_heads,max_unroll=40,drop_prob=0.1):
		super().__init__()
		self.tos_embedding=nn.Embedding(output_dim,hid_dim)
		self.pos_embedding=nn.Embedding(output_dim,hid_dim)

		self.layers=nn.ModuleList([decoder_layer(hid_dim,n_heads,pf_dim,drop_prob) for _ in range(n_layers)])

		self.fc_out=nn.Linear(hid_dim,output_dim)
		self.sample = False

		# self.fc_out = nn.Linear(hid_dim, output_dim)
		self.dropout=nn.Dropout(p=drop_prob)
		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()
		self.max_unroll=max_unroll
		self.pad_ind=0

	def init_token(self, batch_size, sos_id=8):
		"""Get Variable of <SOS> Index (batch_size)"""
		# x = torch.LongTensor([sos_id] * batch_size).cuda().unsqueeze(1)
		x =([sos_id] * batch_size)
		return x

	# def trg_mask(self,x):
	# 	tgt_words = x.unsqueeze(2)[:, :, 0]
	# 	tgt_batch, tgt_len = tgt_words.size()
	# 	trg_pad_mask = tgt_words.data.eq(self.pad_ind).unsqueeze(1)
	# 	trg_pad_mask=trg_pad_mask.unsqueeze(2)
	# 	#trg_pad_mask>[batch_size,1,1,trg_len]
	# 	trg_len=x.shape[1]
	# 	trg_sub_mask=torch.tril(torch.ones((trg_len,trg_len)))
	# 	#trg_sub_mask>>[trg_len,trg_len]

	# 	trg_pad_mask=torch.gt(trg_pad_mask,0).cuda()
	# 	trg_sub_mask=torch.gt(trg_sub_mask,0).cuda()

	# 	trg_mask=trg_pad_mask & trg_sub_mask
	# 	#print('trg_mask',trg_mask)
	# 	trg_mask=~trg_mask
		
	# 	#print('reverse_trg_mask',trg_mask)
	# 	#trg_mask>[batch_size,1,trg_len,trg_len]
	# 	return trg_mask

	def trg_mask(self,trg):
		#trg>>[batch_size,trg_len]
		trg_pad_mask=(trg!=self.pad_ind).unsqueeze(1).unsqueeze(2)
		#trg_pad_mask>[batch_size,1,1,trg_len]
		trg_len=trg.shape[1]
		trg_sub_mask=torch.tril(torch.ones((trg_len,trg_len)).cuda()).type(torch.uint8)
		#trg_sub_mask>>[trg_len,trg_len]
		trg_mask=trg_pad_mask & trg_sub_mask
		#trg_mask>[batch_size,1,trg_len,trg_len]
		return trg_mask

	def decode(self, out):
		"""
		Args:
			out: unnormalized word distribution [batch_size, vocab_size]
		Return:
			x: word_index [batch_size]
		"""

		# Sample next word from multinomial word distribution
		
		if self.sample:
			# x: [batch_size] - word index (next input)
			x = torch.multinomial(self.softmax(out / self.temperature), 2).view(-1)
		# Greedy sampling
		else:
			# x: [batch_size] - word index (next input)
			_, x = out.max(dim=2)
		return x[:,-1].item()
		
		#return out.argmax(2)[:,-1].unsqueeze(1).unsqueeze(2)

	def forward_step(self, enc_src, trg, src_mask):
		trg_mask=self.trg_mask(trg)
		# print(trg_mask[3])
		# print(trg_mask.size())
		# print(trg_mask[0])
		batch_size=trg.shape[0]
		trg_len=trg.shape[1]

		pos=torch.arange(0,trg_len).unsqueeze(0).repeat(batch_size,1).cuda()
		pos_embed=self.pos_embedding(pos)
		tos_embed=self.tos_embedding(trg).cuda()
		trg=self.dropout(tos_embed*self.scale+pos_embed)
		##trg>>[batch_size,trg_len,hid_dim]
		
		for layer in self.layers:
			output,attention=layer(enc_src,trg,src_mask,trg_mask)
		output=self.fc_out(output)

		#output>>[batch_size,trg_len,output_dim]
		return output,attention

	def forward(self, enc_src, trg, src_mask, sos_id, decode=False):
		##enc_src>>[batch_size,src_len]
		##src_mask>>[batch_size,1,1,src_len]
		##trg_mask>>[batch_size,1,trg_len,trg_len]
		##trg>>[batch_size,trg_len]

		batch_size=enc_src.shape[0]
		trg_indexes = self.init_token(batch_size,sos_id)
		# print(trg[0])
		# print(trg_indexes.size())
		#x>[batch_size]
		#x=self.init_token(batch_size,sos_id)
		#x=x.unsqueeze(1).cuda()
	
		if not decode:
			output, attention = self.forward_step(enc_src, trg, src_mask)
			return output, attention
		else:
			# trg_indexes=[]
			
			for i in range(self.max_unroll):
				x = torch.LongTensor(trg_indexes).unsqueeze(0).cuda()

				with torch.no_grad():
					output, attention = self.forward_step(enc_src, x, src_mask)

				# print(output[0][0])
				pred_token = self.decode(output)
				trg_indexes.append(pred_token)

				if pred_token == 3:
					break
			return trg_indexes[1:]

class MTrans_kb_Encoder(nn.Module):

	def __init__(self,input_dim,hid_dim,n_layers,n_heads,pf_dim,drop_prob=0.1):

		super(MTrans_kb_Encoder, self).__init__()
		self.n_layers=n_layers
		self.tos_embed=nn.Embedding(input_dim,hid_dim)

		self.pos_embed=nn.Embedding(input_dim,hid_dim)

		self.mtrans_kb_encoder_layer = nn.ModuleList(
			[MTrans_kb_encoder_layer(hid_dim, pf_dim, n_heads, drop_prob)
			 for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)
		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()

	def forward(self, src, src_mask, history_bank , his_mask, kb_bank,kb_mask):
		""" See :obj:`EncoderBase.forward()`"""

		##src>>[batch_size,src_len]
		##src_mask>>[batch_size,1,1,src_len]
		batch_size=src.shape[0]
		src_len=src.shape[1]
		pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).cuda()
		pos_embedding = self.pos_embed(pos) ##[batch_size,src_len,hid_dim]
		token_embedding = self.tos_embed(src) ##[batch_size,src_len,hid_dim]
		src=self.dropout(token_embedding*(self.scale)+pos_embedding)

		# Run the forward pass of every layer of the tranformer.
		for i in range(self.n_layers):
			out = self.mtrans_kb_encoder_layer[i](src, src_mask, history_bank, his_mask, kb_bank, kb_mask)
		out = self.layer_norm(out)

		return token_embedding, out.contiguous(), src_mask

class MTrans_aspect_Encoder(nn.Module):

	def __init__(self,input_dim,hid_dim,n_layers,n_heads,pf_dim,drop_prob=0.1):

		super(MTrans_aspect_Encoder, self).__init__()
		self.n_layers=n_layers
		self.tos_embed=nn.Embedding(input_dim,hid_dim)

		self.pos_embed=nn.Embedding(input_dim,hid_dim)

		self.mtrans_aspect_encoder_layer = nn.ModuleList(
			[MTrans_aspect_encoder_layer(hid_dim, pf_dim, n_heads, drop_prob)
			 for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)
		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()

	def forward(self, src, src_mask, history_bank , his_mask, aspect_bank, aspect_mask):
		""" See :obj:`EncoderBase.forward()`"""

		##src>>[batch_size,src_len]
		##src_mask>>[batch_size,1,1,src_len]
		batch_size=src.shape[0]
		src_len=src.shape[1]
		pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).cuda()
		pos_embedding = self.pos_embed(pos) ##[batch_size,src_len,hid_dim]
		token_embedding = self.tos_embed(src) ##[batch_size,src_len,hid_dim]
		src=self.dropout(token_embedding*(self.scale)+pos_embedding)

		# Run the forward pass of every layer of the tranformer.
		for i in range(self.n_layers):
			out = self.mtrans_aspect_encoder_layer[i](src, src_mask, history_bank, his_mask, aspect_bank, aspect_mask)
		out = self.layer_norm(out)

		return token_embedding, out.contiguous(), src_mask

class MTrans_img_kb_Encoder(nn.Module):

	def __init__(self,input_dim,hid_dim,n_layers,n_heads,pf_dim,drop_prob=0.1):

		super(MTrans_img_kb_Encoder, self).__init__()
		self.n_layers=n_layers
		self.tos_embed=nn.Embedding(input_dim,hid_dim)

		self.pos_embed=nn.Embedding(input_dim,hid_dim)

		self.mtrans_img_kb_encoder_layer = nn.ModuleList(
			[MTrans_img_kb_encoder_layer(hid_dim, pf_dim, n_heads, drop_prob)
			 for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)
		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()

	def forward(self, src, src_mask, history_bank , img_bank, img_mask, his_mask, kb_bank,kb_mask):
		""" See :obj:`EncoderBase.forward()`"""

		##src>>[batch_size,src_len]
		##src_mask>>[batch_size,1,1,src_len]
		batch_size=src.shape[0]
		src_len=src.shape[1]
		pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).cuda()
		pos_embedding = self.pos_embed(pos) ##[batch_size,src_len,hid_dim]
		token_embedding = self.tos_embed(src) ##[batch_size,src_len,hid_dim]
		src=self.dropout(token_embedding*(self.scale)+pos_embedding)

		# Run the forward pass of every layer of the tranformer.
		for i in range(self.n_layers):
			out = self.mtrans_img_kb_encoder_layer[i](src, src_mask, img_bank, img_mask, history_bank, his_mask, kb_bank, kb_mask)
		out = self.layer_norm(out)

		return token_embedding, out.contiguous(), src_mask

class MTrans_kb_aspect_Encoder(nn.Module):

	def __init__(self,input_dim,hid_dim,n_layers,n_heads,pf_dim,drop_prob=0.1):

		super(MTrans_kb_aspect_Encoder, self).__init__()
		self.n_layers=n_layers
		self.tos_embed=nn.Embedding(input_dim,hid_dim)

		self.pos_embed=nn.Embedding(input_dim,hid_dim)

		self.mtrans_kb_aspect_encoder_layer = nn.ModuleList(
			[MTrans_kb_aspect_encoder_layer(hid_dim, pf_dim, n_heads, drop_prob)
			 for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)
		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()

	def forward(self, src, src_mask, history_bank , aspect_bank, aspect_mask, his_mask, kb_bank,kb_mask):
		""" See :obj:`EncoderBase.forward()`"""

		##src>>[batch_size,src_len]
		##src_mask>>[batch_size,1,1,src_len]
		batch_size=src.shape[0]
		src_len=src.shape[1]
		pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).cuda()
		pos_embedding = self.pos_embed(pos) ##[batch_size,src_len,hid_dim]
		token_embedding = self.tos_embed(src) ##[batch_size,src_len,hid_dim]
		src=self.dropout(token_embedding*(self.scale)+pos_embedding)

		# Run the forward pass of every layer of the tranformer.
		for i in range(self.n_layers):
			out = self.mtrans_kb_aspect_encoder_layer[i](src, src_mask, aspect_bank, aspect_mask, history_bank, his_mask, kb_bank, kb_mask)
		out = self.layer_norm(out)

		return token_embedding, out.contiguous(), src_mask

class MTrans_img_Encoder(nn.Module):

	def __init__(self,input_dim,hid_dim,n_layers,n_heads,pf_dim,drop_prob=0.1):

		super(MTrans_img_Encoder, self).__init__()
		self.n_layers=n_layers
		self.tos_embed=nn.Embedding(input_dim,hid_dim)

		self.pos_embed=nn.Embedding(input_dim,hid_dim)

		self.mtrans_img_encoder_layer = nn.ModuleList(
			[MTrans_img_encoder_layer(hid_dim, pf_dim, n_heads, drop_prob)
			 for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)
		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()

	def forward(self, src, src_mask, history_bank , img_bank, img_mask, his_mask):
		""" See :obj:`EncoderBase.forward()`"""

		##src>>[batch_size,src_len]
		##src_mask>>[batch_size,1,1,src_len]
		batch_size=src.shape[0]
		src_len=src.shape[1]
		pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).cuda()
		pos_embedding = self.pos_embed(pos) ##[batch_size,src_len,hid_dim]
		token_embedding = self.tos_embed(src) ##[batch_size,src_len,hid_dim]
		src=self.dropout(token_embedding*(self.scale)+pos_embedding)

		# Run the forward pass of every layer of the tranformer.
		for i in range(self.n_layers):
			out = self.mtrans_img_encoder_layer[i](src, src_mask, img_bank, img_mask, history_bank, his_mask)
		out = self.layer_norm(out)

		return token_embedding, out.contiguous(), src_mask

class MTrans_img_aspect_Encoder(nn.Module):

	def __init__(self,input_dim,hid_dim,n_layers,n_heads,pf_dim,drop_prob=0.1):

		super(MTrans_img_aspect_Encoder, self).__init__()
		self.n_layers=n_layers
		self.tos_embed=nn.Embedding(input_dim,hid_dim)

		self.pos_embed=nn.Embedding(input_dim,hid_dim)

		self.mtrans_img_aspect_encoder_layer = nn.ModuleList(
			[MTrans_img_aspect_encoder_layer(hid_dim, pf_dim, n_heads, drop_prob)
			 for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)
		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()

	def forward(self, src, src_mask, history_bank , img_bank, img_mask, his_mask, aspect_bank, aspect_mask):
		""" See :obj:`EncoderBase.forward()`"""

		##src>>[batch_size,src_len]
		##src_mask>>[batch_size,1,1,src_len]
		batch_size=src.shape[0]
		src_len=src.shape[1]
		pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).cuda()
		pos_embedding = self.pos_embed(pos) ##[batch_size,src_len,hid_dim]
		token_embedding = self.tos_embed(src) ##[batch_size,src_len,hid_dim]
		src=self.dropout(token_embedding*(self.scale)+pos_embedding)

		# Run the forward pass of every layer of the tranformer.
		for i in range(self.n_layers):
			out = self.mtrans_img_aspect_encoder_layer[i](src, src_mask, img_bank, img_mask, history_bank, his_mask, 
				                                                                           aspect_bank, aspect_mask)
		out = self.layer_norm(out)

		return token_embedding, out.contiguous(), src_mask

class MTrans_img_kb_aspect_Encoder(nn.Module):

	def __init__(self,input_dim,hid_dim,n_layers,n_heads,pf_dim,drop_prob=0.1):

		super(MTrans_img_kb_aspect_Encoder, self).__init__()
		self.n_layers=n_layers
		self.tos_embed=nn.Embedding(input_dim,hid_dim)

		self.pos_embed=nn.Embedding(input_dim,hid_dim)

		self.mtrans_img_kb_aspect_encoder_layer = nn.ModuleList(
			[MTrans_img_kb_aspect_encoder_layer(hid_dim, pf_dim, n_heads, drop_prob)
			 for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)
		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()

	def forward(self, src, src_mask, history_bank , img_bank, img_mask, his_mask, kb_bank,kb_mask,aspect_bank, aspect_mask):
		""" See :obj:`EncoderBase.forward()`"""

		##src>>[batch_size,src_len]
		##src_mask>>[batch_size,1,1,src_len]
		batch_size=src.shape[0]
		src_len=src.shape[1]
		pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).cuda()
		pos_embedding = self.pos_embed(pos) ##[batch_size,src_len,hid_dim]
		token_embedding = self.tos_embed(src) ##[batch_size,src_len,hid_dim]
		src=self.dropout(token_embedding*(self.scale)+pos_embedding)

		# Run the forward pass of every layer of the tranformer.
		for i in range(self.n_layers):
			out = self.mtrans_img_kb_aspect_encoder_layer[i](src, src_mask, img_bank, img_mask, history_bank, his_mask, kb_bank, kb_mask,
				                                                                                        aspect_bank, aspect_mask)
		out = self.layer_norm(out)

		return token_embedding, out.contiguous(), src_mask