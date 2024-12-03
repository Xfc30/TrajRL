from math import ceil

from libcity.model.layers.nodeembedding import NodeEmbedding
from libcity.model.layers.pos_encoding import *
from libcity.model.layers.basics import *
from libcity.model.layers.attention import *
from libcity.model.layers.patch_mask import *
from libcity.model.layers.sequencepooler import *


class TrajRLContrastiveLM(nn.Module):
    def __init__(self, config, data_feature):
        super().__init__()
        self.config = config


        self.trajrl = TrajRL(config, data_feature)

    def forward(self, contra_view1, contra_view2, masked_input, padding_masks,
                batch_temporal_mat, padding_masks1=None, padding_masks2=None,
                batch_temporal_mat1=None, batch_temporal_mat2=None,
                graph_dict=None):
        """
        Args:
            contra_view1: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            contra_view2: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            masked_input: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
            graph_dict(dict):
        Returns:
            xb: [bs x d_model ]
        """
        if padding_masks1 is None:
            padding_masks1 = padding_masks
        if padding_masks2 is None:
            padding_masks2 = padding_masks
        if batch_temporal_mat1 is None:
            batch_temporal_mat1 = batch_temporal_mat
        if batch_temporal_mat2 is None:
            batch_temporal_mat2 = batch_temporal_mat
        out_view1, _ ,_= self.trajrl(contra_view1,
                                                   temporal_mat=batch_temporal_mat1, graph_dict=graph_dict, mlm=True)
        out_view2, _ ,_= self.trajrl(contra_view2,
                                                   temporal_mat=batch_temporal_mat2, graph_dict=graph_dict, mlm=True)
        _, a ,mask= self.trajrl(masked_input, temporal_mat=batch_temporal_mat,
                                           graph_dict=graph_dict, mlm=True)
        # xb = self.pooler(xb)
        return out_view1, out_view2, a,mask


class TrajRL(nn.Module):
    def __init__(self, config, data_feature):
        super().__init__()
        self.config = config

        self.vocab_size = data_feature.get('vocab_size')
        self.usr_num = data_feature.get('usr_num')
        self.node_fea_dim = data_feature.get('node_fea_dim')

        # self.patch_len = data_feature.get('patch_len')
        # self.stride = data_feature.get('stride')
        # self.num_patch = data_feature.get('num_patch')
        self.win_size = data_feature.get('win_size', 2)
        self.mask_ratio = data_feature.get('mask_ratio')

        self.seq_len = self.config.get('seq_len', 128)
        self.e_blocks = self.config.get('num_blocks', 8)
        self.d_model = self.config.get('d_model', 768)
        self.dropout = self.config.get('dropout', 0.1)
        self.add_time_in_day = self.config.get('add_time_in_day', True)
        self.add_day_in_week = self.config.get('add_day_in_week', True)
        self.add_time_interval = self.config.get('add_time_interval', True)
        self.max_time_scale_s = self.config.get('max_time_scale_s', 3000)
        self.add_gat = self.config.get('add_gat', False)
        self.add_hgnn = self.config.get('add_hgnn', False)
        self.time_interval_scale_list = self.config.get('time_interval_scales', [10])
        self.gat_heads_per_layer = self.config.get('gat_heads_per_layer', [8, 1])
        self.gat_features_per_layer = self.config.get('gat_features_per_layer', [16, self.d_model])
        self.gat_dropout = self.config.get('gat_dropout', 0.6)
        self.gat_avg_last = self.config.get('gat_avg_last', True)
        self.load_trans_prob = self.config.get('load_trans_prob', False)
        self.with_mask = self.config.get('with_mask', False)

        self.embedding = NodeEmbedding(d_model=self.d_model, dropout=self.dropout,
                                       add_time_in_day=self.add_time_in_day, add_day_in_week=self.add_day_in_week,
                                       add_time_interval=self.add_time_interval,
                                       max_time_scale_s=self.max_time_scale_s,
                                       time_interval_scale_list=self.time_interval_scale_list,
                                       add_pe=True, node_fea_dim=self.node_fea_dim, add_gat=self.add_gat,
                                       add_hgnn=self.add_hgnn,
                                       gat_heads_per_layer=self.gat_heads_per_layer,
                                       gat_features_per_layer=self.gat_features_per_layer,
                                       gat_dropout=self.gat_dropout,
                                       load_trans_prob=self.load_trans_prob, avg_last=self.gat_avg_last)

        if self.with_mask:
            self.patcher = PatchMasker(self.win_size, self.win_size, self.mask_ratio)
            self.mask_l = MaskedLanguageModel(self.d_model, self.vocab_size)
        self.patchtst_layers = nn.ModuleList()
        self.patchtst_layers.append(scale_block(c_in=1, fea_dim=self.d_model, patch_len=1, stride=1,
                                                mask_ratio=self.mask_ratio, num_patch=self.seq_len,
                                                d_model=self.d_model,
                                                usr_num=self.usr_num, with_mask=self.with_mask))

        for i in range(1, self.e_blocks):
            self.patchtst_layers.append(scale_block(c_in=1, fea_dim=self.d_model, patch_len=self.win_size, stride=self.win_size,
                                                    num_patch=ceil(self.seq_len / (self.win_size ** i)),
                                                    d_model=self.d_model,
                                                    usr_num=self.usr_num))
        self.pooler_blocks = nn.ModuleList()
        for i in range(0, self.e_blocks):
            self.pooler_blocks.append(SequenceMeanPoolLayer(self.d_model, self.d_model))

        self.weighted_average = WeightedAverage(n=self.e_blocks, d_model=self.d_model)

    def forward(self, x, temporal_mat=None, graph_dict=None,mlm=False):
        if self.with_mask and mlm:
            z = x.unsqueeze(-1)
            zb_mask, mask = self.patcher.patch_masking(z)
            z =torch.flatten(z.squeeze(2))
            embedding_output = self.embedding(sequence=z, batch_temporal_mat_list=temporal_mat, graph_dict=graph_dict)

        embedding_output = self.embedding(sequence=x, batch_temporal_mat_list=temporal_mat, graph_dict=graph_dict)
        # road_sequence_fea = embedding_output

        x_embs = []  # list of tensor (batch_size, num_patch, d_model)
        intput = embedding_output
        for block in self.patchtst_layers:
            x_lay_emb, _, _ = block(intput, x)
            intput = x_lay_emb.unsqueeze(2)
            x_embs.append(x_lay_emb)
        if self.with_mask and mlm:
            a = self.mask_l(x_embs[0])
        for i in range(len(self.pooler_blocks)):
            x_embs[i] = self.pooler_blocks[i](x_embs[i])  # list of tensor (batch_size, d_model) num_patch:128,64,32...1
        x_emb = self.weighted_average(x_embs)
        if self.with_mask and mlm:
            return x_emb,a,mask
        else:
            return x_emb


class scale_block(nn.Module):
    """
    PatchTST with mask (for self-supervised) or not
    """

    # 修改为单变量时间序列
    def __init__(self, patch_len: int, stride: int, mask_ratio: float, num_patch: int,
                 n_layers: int = 3, fea_dim=64, d_model=128, n_heads=16, shared_embedding=True, d_ff: int = 256,
                 attn_dropout: float = 0.2, dropout: float = 0.2, act: str = "gelu",
                 res_attention: bool = True, pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True, head_dropout=0.2, usr_num: int = 100,
                 with_mask: bool = False, verbose: bool = False, **kwargs):
        super().__init__()
        self.usr_num = usr_num
        self.with_mask = with_mask
        # Backbone
        self.backbone = Encoder(num_patch=num_patch, patch_len=patch_len,
                                n_layers=n_layers, fea_dim=fea_dim, d_model=d_model, n_heads=n_heads,
                                shared_embedding=shared_embedding, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act,
                                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        self.num_patch = num_patch

        self.patcher = Patcher(patch_len, stride)

    def forward(self, z_emb):
        """
        z_emb: tensor [bs x seq_len x embedding_size] now embedding_size = d_model
        Returns:
            zb_patch: [bs x num_patch x d_model]
            zb_patch_label: [bs x num_patch x patch_len x d_model]
            mask: [bs x num_patch]
        """

        # zb_patch: tensor [bs x num_patch x 1 x patch_len x embedding_size]
        zb_patch = self.patcher.patch(z_emb)  #
        zb_patch = zb_patch.sum(dim=2)

        zb_patch = self.backbone(zb_patch)  # [bs x num_patch x d_model]

        return zb_patch


class Encoder(nn.Module):
    def __init__(self, num_patch, patch_len,
                 n_layers=3, fea_dim=64, d_model=128, n_heads=16,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):

        super().__init__()
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model

        self.gru = PatchGRUEncoder(fea_dim, d_model, num_layers=2)

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = CrossEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                    pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers,
                                    store_attn=store_attn)

    def forward(self, x) -> Tensor:
        """
        x: tensor [bs x num_patch x patch_len x d_model]
        """
        bs = x.shape[0]
        # num_patch = self.num_patch
        # Input encoding


        x = self.gru(x)  # x: [bs x num_patch x d_model]
        # x = x.transpose(1, 2)  # x: [bs x num_patch x d_model]

        u = torch.reshape(x, (bs, self.num_patch, self.d_model))  # u: [bs x num_patch x d_model]
        u = self.dropout(u + self.W_pos)  # u: [bs * nvars x num_patch x d_model]

        # Encoder
        z = self.encoder(u)  # z: [bs x num_patch x d_model]
        z = torch.reshape(z, (-1, self.num_patch, self.d_model))  # z: [bs x num_patch x d_model]

        return z

class CrossEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                                                             attn_dropout=attn_dropout, dropout=dropout,
                                                             activation=activation, res_attention=res_attention,
                                                             pre_norm=pre_norm, store_attn=store_attn) for i in
                                     range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor):
        """
        src: tensor [bs x q_len x d_model]
        """
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores)
            return output
        else:
            for mod in self.layers: output = mod(output)
            return output
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True,
                 activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout,
                                            res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor] = None):
        """
        src: tensor [bs x q_len x d_model]
        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev)
        else:
            src2, attn = self.self_attn(src, src, src)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


class PatchGRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False):
        super(PatchGRUEncoder, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        # self.device = device
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        if self.bidirectional:
            self.linear = nn.Linear(hidden_size * 2, hidden_size,bias=False)

    def forward(self, x):
        batch_size, num_patch, patch_len, d_model = x.size()

        # Reshape to (batch_size * num_patch, patch_len, d_model)
        x = x.reshape(-1, patch_len, d_model)

        # Initialize hidden state
        h0 = torch.zeros(self.gru.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)

        # GRU forward
        out, _ = self.gru(x, h0)

        # Pooling: average pooling over the time dimension (patch_len)
        out = torch.mean(out, dim=1)  # Shape: (batch_size * num_patch, hidden_size)

        # Reshape back to (batch_size, num_patch, hidden_size)
        out = out.reshape(batch_size, num_patch, self.hidden_size * (2 if self.bidirectional else 1))
        if self.bidirectional:
            out = self.linear(out)
        return out

class MaskedLanguageModel(nn.Module):


    def __init__(self, hidden, vocab_size):

        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
