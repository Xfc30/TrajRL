from math import ceil

import torch
from torch import nn
import torch.nn.functional as F
import dhg


class GAT2(nn.Module):

    def __init__(self, d_model, in_feature, num_heads_per_layer, num_features_per_layer,
                 add_skip_connection=True, bias=True, dropout=0.6, load_trans_prob=True, avg_last=True):
        super().__init__()
        self.d_model = d_model
        assert len(num_heads_per_layer) == len(num_features_per_layer), f'Enter valid arch params.'

        num_features_per_layer = [in_feature] + num_features_per_layer
        num_heads_per_layer = [1] + num_heads_per_layer  # trick - so that I can nicely create GAT layers below
        if avg_last:
            assert num_features_per_layer[-1] == d_model
        else:
            assert num_features_per_layer[-1] * num_heads_per_layer[-1] == d_model
        num_of_layers = len(num_heads_per_layer) - 1

        gat_layers = []  # collect GAT layers
        for i in range(num_of_layers):
            if i == num_of_layers - 1:
                if avg_last:
                    concat_input = False
                else:
                    concat_input = True
            else:
                concat_input = True
            layer = GATLayerImp3(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  # consequence of concatenation
                num_out_features=num_features_per_layer[i + 1],
                num_of_heads=num_heads_per_layer[i + 1],
                concat=concat_input,  # last GAT layer does mean avg, the others do concat
                activation=nn.ELU() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                load_trans_prob=load_trans_prob
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )

    def forward(self, node_features, edge_index_input, edge_prob_input, x, return_first_layer=False):
        """

        Args:
            node_features: (vocab_size, fea_dim)
            edge_index_input: (2, E)
            edge_prob_input: (E, 1)
            x: (B, T)
            return_first_layer: bool : return node_fea_emb in first layer gat or not
        Returns:
            (B, T, d_model)

        """
        data = (node_features, edge_index_input, edge_prob_input)
        first_layer_output = None
        # (node_fea_emb, edge_index, edge_prob) = self.gat_net(data)  # (vocab_size, num_channels[-1]), (2, E)
        for i, layer in enumerate(self.gat_net):
            data = layer(data)
            if i == 0 and return_first_layer:
                first_layer_output = data[0]  # Capture the node_fea_emb of the first layer
        node_fea_emb = data[0]
        batch_size, seq_len = x.shape
        node_fea_emb = node_fea_emb.expand((batch_size, -1, -1))  # (B, vocab_size, d_model)
        node_fea_emb = node_fea_emb.reshape(-1, self.d_model)  # (B * vocab_size, d_model)
        x = x.reshape(-1, 1).squeeze(1)  # (B * T,)
        out_node_fea_emb = node_fea_emb[x].reshape(batch_size, seq_len, self.d_model)  # (B, T, d_model)
        if return_first_layer:
            return out_node_fea_emb, first_layer_output
        else:
            return out_node_fea_emb  # (B, T, d_model)



class NodeEmbedding(nn.Module):

    def __init__(self, d_model, add_time_in_day=False, add_day_in_week=False,
                 add_pe=True, add_time_interval=True, max_time_scale_s=5000, time_interval_scale_list=[10],
                 node_fea_dim=10, add_gat=True,
                 gat_heads_per_layer=None, gat_features_per_layer=None, gat_dropout=0.6,
                 load_trans_prob=True, avg_last=True):
        """

        Args:
            vocab_size: total vocab size
            d_model: embedding size of token embedding
            dropout: dropout rate
        """
        super().__init__()
        self.add_min_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week
        self.add_time_interval = add_time_interval
        self.add_pe = add_pe
        self.add_gat = add_gat
        self.add_hgnn = add_hgnn

        if self.add_gat:
            self.token_embedding = GAT2(d_model=d_model, in_feature=node_fea_dim,
                                        num_heads_per_layer=gat_heads_per_layer,
                                        num_features_per_layer=gat_features_per_layer,
                                        add_skip_connection=True, bias=True, dropout=gat_dropout,
                                        load_trans_prob=load_trans_prob, avg_last=avg_last)
        if self.add_pe:
            self.position_embedding = PositionalEmbedding(d_model=d_model)
        if self.add_min_in_day:
            self.day_min_embedding = nn.Embedding(1441, d_model, padding_idx=0)
        if self.add_hour_in_day:
            self.day_hour_embedding = nn.Embedding(25, d_model, padding_idx=0)
        if self.add_day_in_week:
            self.weekday_embedding = nn.Embedding(8, d_model, padding_idx=0)
        if self.add_time_interval:
            if len(time_interval_scale_list)>1:
                self.time_interval_weights = nn.Parameter(torch.randn(len(time_interval_scale_list)))
                with torch.no_grad():
                    self.time_interval_weights.copy_(torch.tensor((0.2, 0.6, 0.2)))
            else:
                self.time_interval_weights = torch.tensor([1.0])
            # 初始化权重
            self.time_interval_embedding = nn.ModuleList(
                [nn.Embedding(ceil(max_time_scale_s / scale)+1, d_model, padding_idx=0) for scale in
                 time_interval_scale_list])

        self.d_model = d_model

    def forward(self, sequence, batch_temporal_mat_list=None, position_ids=None, graph_dict=None,
                return_first_layer=False):
        """

        Args:
            sequence: (B, T, F) [loc, ts, mins, weeks, usr]
            batch_temporal_mat: (B, T)
            position_ids: (B, T) or None
            graph_dict(dict): including:
                in_lap_mx: (vocab_size, lap_dim)
                out_lap_mx: (vocab_size, lap_dim)
                indegree: (vocab_size, )
                outdegree: (vocab_size, )
                return_first_layer: bool : return node_fea_emb in first layer gat or not
        Returns:
            (B, T, d_model)

        """
        if self.add_gat:
            if return_first_layer:
                x, node_fea_first_layer = self.token_embedding(node_features=graph_dict['node_features'],
                                                               edge_index_input=graph_dict['edge_index'],
                                                               edge_prob_input=graph_dict['loc_trans_prob'],
                                                               x=sequence[:, :, 0],
                                                               return_first_layer=return_first_layer)  # (B, T, d_model)
            else:
                x = self.token_embedding(node_features=graph_dict['node_features'],
                                         edge_index_input=graph_dict['edge_index'],
                                         edge_prob_input=graph_dict['loc_trans_prob'],
                                         x=sequence[:, :, 0],
                                         return_first_layer=return_first_layer)  # (B, T, d_model)

        if self.add_pe:
            x += self.position_embedding(x, position_ids)  # (B, T, d_model)
        if self.add_min_in_day:
            x += self.day_min_embedding(sequence[:, :, 2])  # (B, T, d_model)
        if self.add_hour_in_day:
            x += self.day_hour_embedding(ceil(sequence[:, :, 2] / 60))  # (B, T, d_model)
        if self.add_day_in_week:
            x += self.weekday_embedding(sequence[:, :, 3])  # (B, T, d_model)
        if self.add_time_interval:

            weights = F.softmax(self.time_interval_weights, dim=0)
            time_interval_emb_list = []
            for i in range(len(self.time_interval_embedding)):
                time_interval_emb_list.append(self.time_interval_embedding[i](batch_temporal_mat_list[i]))
            sum_time_interval_emb = sum(w * emb for w, emb in zip(weights, time_interval_emb_list))
            x += sum_time_interval_emb  # (B, T, d_model)


        return x


class GATLayerImp3(GATLayer):
    """
    Implementation #3 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric

    But, it's hopefully much more readable! (and of similar performance)

    It's suitable for both transductive and inductive settings. In the inductive setting we just merge the graphs
    into a single graph with multiple components and this layer is agnostic to that fact! <3

    """

    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    nodes_dim = 0      # node dimension/axis
    head_dim = 1       # attention head dimension/axis

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, load_trans_prob=True):

        # Delegate initialization to the base class
        super().__init__(num_in_features, num_out_features, num_of_heads, concat, activation, dropout_prob,
                         add_skip_connection, bias, load_trans_prob)

    def forward(self, data):
        #
        # Step 1: Linear Projection + regularization
        #
        in_nodes_features, edge_index, edge_prob = data  # unpack data edge_prob=(E, 1)
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        # Note: for Cora features are already super sparse so it's questionable how much this actually helps
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

        if self.load_trans_prob:
            # shape = (E, 1) * (1, NH*FOUT) -> (E, NH, FOUT) where NH - number of heads, FOUT - num of output features
            trans_prob_proj = self.linear_proj_tran_prob(edge_prob).view(
                -1, self.num_of_heads, self.num_out_features)  # (E, NH, FOUT)
            trans_prob_proj = self.dropout(trans_prob_proj)
        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, FOUT) -> (N, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(
            scores_source, scores_target, nodes_features_proj, edge_index)
        if self.load_trans_prob:
            # shape = (E, NH, FOUT) * (1, NH, FOUT) -> (E, NH, FOUT) -> (E, NH)
            scores_trans_prob = (trans_prob_proj * self.scoring_trans_prob).sum(dim=-1)  # (E, NH)
            scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted + scores_trans_prob)
        else:
            scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        # shape = (E, NH, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
        # Add stochasticity to neighborhood aggregation
        attentions_per_edge = self.dropout(attentions_per_edge)

        #
        # Step 3: Neighborhood aggregation
        #

        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

        # This part sums up weighted and projected neighborhood feature vectors for every target node
        # shape = (N, NH, FOUT)
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)

        #
        # Step 4: Residual/skip connections, concat and bias
        #
        out_nodes_features = self.skip_concat_bias(in_nodes_features, out_nodes_features)
        return (out_nodes_features, edge_index, edge_prob)

    #
    # Helper functions (without comments there is very little code so don't be scared!)
    #

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """
        As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
        Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
        into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
        in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
        (where 1-3 is overloaded notation it represents the edge 1-3 and it's (exp) score) and similarly for 2-3 and 3-3
         i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.

        Note:
        Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
        and it's a fairly common "trick" used in pretty much every deep learning libcity.
        Check out this link for more details:

        https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning

        """
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_edge = scores_per_edge - scores_per_edge.max()  # (E, NH)
        exp_scores_per_edge = scores_per_edge.exp()  # softmax, (E, NH)

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).

        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # E -> (E, NH)
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)

class GATLayer(torch.nn.Module):
    """
    Base class for all implementations as there is much code that would otherwise be copy/pasted.

    """

    head_dim = 1

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, load_trans_prob=True):

        super().__init__()

        # Saving these as we'll need them in forward propagation in children layers (imp1/2/3)
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection
        self.load_trans_prob = load_trans_prob

        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #
        # You can treat this one matrix as num_of_heads independent W matrices
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        if self.load_trans_prob:
            self.linear_proj_tran_prob = nn.Linear(1, num_of_heads * num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.

        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        if self.load_trans_prob:
            self.scoring_trans_prob = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #
        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the log-softmax along the last dimension
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout_prob)

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        if self.load_trans_prob:
            nn.init.xavier_uniform_(self.linear_proj_tran_prob.weight)
            nn.init.xavier_uniform_(self.scoring_trans_prob)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, in_nodes_features, out_nodes_features):

        # if the tensor is not contiguously stored in memory we'll get an error after we try to do certain ops like view
        # only imp1 will enter this one
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)  # (max_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)  # (max_len, d_model/2)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x, position_ids=None):
        """

        Args:
            x: (B, T, d_model)
            position_ids: (B, T) or None

        Returns:
            (1, T, d_model) / (B, T, d_model)

        """
        if position_ids is None:
            return self.pe[:, :x.size(1)].detach()
        else:
            batch_size, seq_len = position_ids.shape
            pe = self.pe[:, :seq_len, :]  # (1, T, d_model)
            pe = pe.expand((position_ids.shape[0], -1, -1))  # (B, T, d_model)
            pe = pe.reshape(-1, self.d_model)  # (B * T, d_model)
            position_ids = position_ids.reshape(-1, 1).squeeze(1)  # (B * T,)
            output_pe = pe[position_ids].reshape(batch_size, seq_len, self.d_model).detach()
            return output_pe
