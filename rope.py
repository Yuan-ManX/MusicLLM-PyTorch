import torch


def build_rope(
    seq_len: int, head_dim: int, base: int = 10000
) -> torch.Tensor:
    """
    构建旋转位置嵌入（Rotary Position Embedding, RoPE）。

    参数:
        seq_len (int): 序列长度，例如1024。
        head_dim (int): 每个注意力头的维度，例如768/24。
        base (int): 基数，默认为10000。

    返回:
        cache (torch.Tensor): 旋转位置嵌入缓存，形状为 (t, head_dim/2, 2)。
    """
    # 计算角度 theta_i
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim)) # shape: (head_dim/2,)
    
    # 生成位置索引张量，形状为 (seq_len,)
    seq_idx = torch.arange(seq_len)

    # 计算位置索引与 $\theta_i$ 的外积，得到每个位置和每个 $\theta_i$ 的乘积
    # 结果 idx_theta 的形状为 (seq_len, head_dim/2)
    idx_theta = torch.outer(seq_idx, theta).float() # shape: (seq_len, head_dim/2)

    # 使用 torch.stack 将余弦和正弦值堆叠在一起，形状为 (seq_len, head_dim/2, 2)
    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1) # shape: (seq_len, head_dim/2, 2)

    # 返回旋转位置嵌入缓存
    return cache


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    """
    应用旋转位置嵌入到输入张量上。

    参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, head_dim)。
        rope_cache (torch.Tensor): 旋转位置嵌入缓存，形状为 (seq_len, head_dim/2, 2)。

    返回:
        x_out2 (torch.Tensor): 应用了旋转位置嵌入后的张量，形状为 (batch_size, seq_len, head_dim)。
    """
    # 获取输入张量的时间步长度 T
    T = x.size(1)
    # 截取 rope_cache 的前 T 个时间步，以匹配输入张量的长度
    rope_cache = rope_cache[:T] # shape: (T, head_dim/2, 2)

    # 将输入张量转换为浮点数类型
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # 形状变为 (batch_size, seq_len, head_dim/2, 2)

    # 重塑 rope_cache 以便与 xshaped 进行广播相乘
    # 形状变为 (1, T, 1, head_dim/2, 2)
    rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)

    # 应用旋转位置嵌入：
    # x_out2 的第一个部分为 xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1]
    # x_out2 的第二个部分为 xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1]
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    ) # 形状为 (batch_size, T, head_dim/2, 2)

    # 将 x_out2 的最后两个维度展平，形状变为 (batch_size, T, head_dim)
    x_out2 = x_out2.flatten(3)

    # 返回应用了旋转位置嵌入后的张量
    return x_out2.type_as(x)
