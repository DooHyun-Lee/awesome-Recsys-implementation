import torch 
import torch.nn as nn
import numpy as np 

class FFWUnoff(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(FFWUnoff, self).__init__()

        self.fc1 = nn.Linear(hidden_units, hidden_units)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # input will given [N, L, E] dimension
        outputs = self.dropout2(self.fc2(self.relu(self.dropout1(self.fc1(x)))))
        outputs += x
        return outputs

class SASRec(nn.Module):
    # ffw with 2-layer mlp 
    def __init__(self, user_total, item_total, device, args):
        super(SASRec, self).__init__()

        self.user_total = user_total
        self.item_total = item_total
        self.device = device

        self.item_emb = nn.Embedding(self.item_total +1, args.hidden_units, padding_idx=0)
        self.position_emb = nn.Embedding(args.maxlen, args.hidden_units)
        # also apply dropout on the embedding
        self.emb_dropout = nn.Dropout(args.dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        # layer-norm to last hidden_units dim
        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)

        for i in range(args.num_blocks):
            attn_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(attn_layernorm)
            attn_layer = nn.MultiheadAttention(embed_dim=args.hidden_units, num_heads=args.num_heads, dropout=args.dropout_rate)
            self.attention_layers.append(attn_layer)

            ffw_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(ffw_layernorm)
            ffw_layer = FFWUnoff(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(ffw_layer)

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_normal_(module.in_proj_weight)
            nn.init.xavier_normal_(module.out_proj.weight)
            if module.in_proj_bias is not None: 
                module.in_proj_bias.data.zero_()
            if module.out_proj.bias is not None:
                module.out_proj.bias.data.zero_()
        # should we init layernorm as well? 

    def seq2feature(self, seqs):
        # seqs : [batch_size, max_len] in numpy
        seqs_feat = self.item_emb(torch.LongTensor(seqs).to(self.device))
        # https://datascience.stackexchange.com/questions/87906/transformer-model-why-are-word-embeddings-scaled-before-adding-positional-encod
        # multiply root(d) to weights of embedding result
        seqs_feat *= (self.item_emb.embedding_dim ** 0.5)
        # create [0, 1, ..., max_len -1] array batch_size times
        position_ids = np.tile(np.array(range(seqs.shape[1])), [seqs.shape[0], 1])
        seqs_feat += self.position_emb(torch.LongTensor(position_ids).to(self.device))
        seqs_feat = self.emb_dropout(seqs_feat) # [batch_size, max_len, args.hidden_units]

        pad_mask = torch.BoolTensor(seqs == 0).to(self.device) # 1 if padding item
        seqs_feat *= ~pad_mask.unsqueeze(-1)

        # attention mask y axis: target sequence, x axis : source sequence
        # true : not attend 
        # target j should attend only source i when j >= i
        # mask : upper triangle without diagonal 
        seq_len = seqs_feat.shape[1]
        attention_mask = ~torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.device))

        for i in range(len(self.attention_layers)):
            seqs_feat = torch.transpose(seqs_feat, 0, 1) # [L, N, E]
            Q = self.attention_layernorms[i](seqs_feat)
            attention_outs, _ = self.attention_layers[i](Q, seqs_feat, seqs_feat, attention_mask)
            seqs_feat = Q + attention_outs
            seqs_feat = torch.transpose(seqs_feat, 0, 1) 

            seqs_feat = self.forward_layernorms[i](seqs_feat)
            seqs_feat = self.forward_layers[i](seqs_feat)
            seqs_feat *= ~pad_mask.unsqueeze(-1) # [batch_size, max_len, args.hidden_units]
        
        log_feat = self.last_layernorm(seqs_feat)
        return log_feat

    def forward(self, user_ids, seq, pos, neg):
        # all inputs in numpy 
        log_feat = self.seq2feature(seq)
        
        pos_emb = self.item_emb(torch.LongTensor(pos).to(self.device))
        neg_emb = self.item_emb(torch.LongTensor(neg).to(self.device))

        pos_logits = (log_feat * pos_emb).sum(dim=-1) # [batch_size, max_len]
        neg_logits = (log_feat * neg_emb).sum(dim=-1)
        return pos_logits, neg_logits