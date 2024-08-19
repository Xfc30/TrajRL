import torch
import torch.nn as nn


class SequenceMeanPoolLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SequenceMeanPoolLayer, self).__init__()
        self.embedding_size = in_dim
        self.d_model = out_dim
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # x: (length, in_dim)
        mean_embedding = torch.mean(x, dim=-2)  # 对序列维度取平均
        output = self.linear(mean_embedding)  # 将平均值投影到维度为 out_dim 的空间
        return output


class WeightedAverage(nn.Module):
    def __init__(self, n, d_model):
        super(WeightedAverage, self).__init__()
        # 初始化可学习权重，初始值为均匀分布
        self.weights = nn.Parameter(torch.ones(n, dtype=torch.float32) / n)
        self.n = n
        self.d_model = d_model

    def forward(self, tensors):
        # 确保输入的张量数量与权重数量一致
        assert len(tensors) == self.n, "张量数量必须与初始化的 n 一致"

        # 将 weights 归一化为和为 1
        weights = torch.softmax(self.weights, dim=0)

        # 初始化加权和为零张量
        weighted_sum = torch.zeros_like(tensors[0])

        # 计算加权和
        for tensor, weight in zip(tensors, weights):
            weighted_sum += tensor * weight

        return weighted_sum


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
        batch_size, num_patch, nvars, patch_len, d_model = x.size()

        # Reshape to (batch_size * num_patch * nvars, patch_len, d_model)
        x = x.reshape(-1, patch_len, d_model)

        # Initialize hidden state
        h0 = torch.zeros(self.gru.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)

        # GRU forward
        out, _ = self.gru(x, h0)

        # Pooling: average pooling over the time dimension (patch_len)
        out = torch.mean(out, dim=1)  # Shape: (batch_size * num_patch * nvars, hidden_size)

        # Reshape back to (batch_size, num_patch, nvars, hidden_size)
        out = out.reshape(batch_size, num_patch, nvars, self.hidden_size * (2 if self.bidirectional else 1))
        if self.bidirectional:
            out = self.linear(out)
        return out


