import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from rope import build_rope, apply_rope


@dataclass
class LlamaConfig:
    """
    LLaMA 模型配置类，用于定义 LLaMA 模型的各种参数。

    Attributes:
        block_size (int, 默认: 2048): 
            输入序列的最大长度。模型在处理输入时，会将输入序列分割成多个块，每个块的长度不超过 block_size。
        vocab_size (int, 默认: 32000): 
            词汇表的大小，即模型能够处理的唯一标记（tokens）的数量。
            通常建议该值能够被64整除，以优化内存对齐和计算效率。
        n_layer (int, 默认: 32): 
            Transformer 模型中编码器或解码器的层数，也称为 Transformer 块的数量。
            层数越多，模型的表达能力通常越强，但计算成本也越高。
        n_head (int, 默认: 32): 
            每个 Transformer 块中注意力机制的头数。
            多头注意力机制允许模型在不同表示子空间上同时关注输入的不同部分。
        n_embd (int, 默认: 4096): 
            嵌入维度（embedding dimension），即每个输入标记被映射到的向量空间的维度。
            嵌入维度越大，模型对输入的表示能力通常越强，但计算和内存需求也越高。
    """
    block_size: int = 2048
    vocab_size: int = 32000  # Better to be divied by 64
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096


# 默认的 LLaMA 配置字典
llama_configs = {
    "7B": dict(n_layer=32, n_head=32, n_embd=4096),
    "13B": dict(n_layer=40, n_head=40, n_embd=5120),
    "30B": dict(n_layer=60, n_head=52, n_embd=6656),
    "65B": dict(n_layer=80, n_head=64, n_embd=8192),
}


class Llama(nn.Module):
    """
    LLaMA 模型类。

    LLaMA（Language Model for Dialogue Applications）是一种基于 Transformer 的语言模型，
    适用于对话系统和其他自然语言处理任务。
    """

    def __init__(self, config: LlamaConfig) -> None:
        """
        初始化 LLaMA 模型。

        Args:
            config (LlamaConfig): LLaMA 模型的配置参数。
        """
        super().__init__()
        # 保存配置参数
        self.config = config

        # 词嵌入层：将词汇表中的单词索引转换为嵌入向量
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Transformer 块列表：构建多个 Transformer 块
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))

        # 输出归一化层：使用 RMSNorm 进行归一化
        self.ln_f = RMSNorm(config.n_embd)

        # 语言模型头：将 Transformer 块的输出映射到词汇表大小，用于预测下一个词的概率分布
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 构建 RoPE（旋转位置嵌入）缓存
        rope = build_rope(
            seq_len=config.block_size,  # 序列长度，例如2048
            head_dim=config.n_embd // config.n_head,  # 每个注意力头的维度，例如4096 // 32 = 128
        )  # 形状为 (t, head_dim/2, 2)

        # 将 rope 注册为模型的缓冲区，不参与梯度更新
        self.register_buffer(name="rope", tensor=rope)

    def _init_weights(self, module: nn.Module, config: LlamaConfig) -> None:
        """
        初始化模型权重。

        Args:
            module (nn.Module): 需要初始化的模型模块。
        """
        if isinstance(module, nn.Linear):
            # 对线性层的权重进行正态分布初始化，均值为0，标准差为 0.02 / sqrt(2 * n_layer)
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        elif isinstance(module, nn.Embedding):
            # 对嵌入层的权重进行正态分布初始化，均值为0，标准差为 0.02 / sqrt(2 * n_layer)
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def forward(
        self, 
        ids: torch.LongTensor,
        mask: None | torch.Tensor = None,
    ) -> torch.Tensor:
        """
        使用 LLaMA 模型进行下一个词预测。

        Args:
            ids (torch.LongTensor): 输入的词 ID 序列，形状为 (batch_size, time_steps)。
            mask (torch.Tensor, 可选): 注意力掩码，形状为 (1, 1, time_steps, time_steps)。默认为 None。

        Returns:
            torch.Tensor: 预测的下一个词的 logits，形状为 (batch_size, time_steps, vocab_size)。
        """
        # 获取设备信息（CPU 或 GPU）
        device = ids.device
        # 获取批处理大小 (B) 和时间步数 (T)
        B, T = ids.shape

        # 确保输入序列长度不超过模型的最大序列长度
        assert T <= self.config.block_size, "Can not forward sequence of {T} > {self.config.block_size}"

        # 如果没有提供掩码，则构建因果掩码（causal mask）
        if mask is None:
            # 构建因果掩码并移动到指定设备
            mask = build_causal_mask(seq_len=T).to(device)

        # 词嵌入：将词 ID 转换为嵌入向量，形状为 (B, T, d)
        x = self.wte(ids)  # shape: (b, t, d)

        # Transformer 层：逐个应用 Transformer 块
        for block in self.blocks:
            # 每个块的输入包括输入张量、RoPE 缓存和掩码
            x = block(x, self.rope, mask) # x 的形状为 (B, T, d)

        # 输出归一化：对 Transformer 块的输出进行归一化，形状为 (B, T, d)
        x = self.ln_f(x) 

        # 语言模型头：将归一化后的输出映射到词汇表大小，生成 logits，形状为 (B, T, v)
        logits = self.lm_head(x) 

        return logits

    @torch.no_grad()
    def generate(
        self, 
        ids: torch.LongTensor, 
        max_new_ids: int, 
        temperature: float = 1.0, 
        top_k: None | int = None
    ):
        """
        使用自回归方式进行下一个 ID 的采样，生成新的文本序列。

        确保在调用此方法之前调用 model.eval() 将模型设置为评估模式。

        参数:
            ids (torch.LongTensor): 输入的词 ID 序列，形状为 (batch_size, 1)。
            max_new_ids (int): 要生成的新的词 ID 的最大数量。
            temperature (float, 可选): 采样温度，控制生成文本的多样性。默认为1.0。
                                        温度越高，生成的文本越多样；温度越低，生成的文本越确定。
            top_k (int, 可选): 在采样前保留的最高 k 个概率的词 ID。默认为 None，表示不进行 top-k 截断。

        返回:
            new_ids (torch.LongTensor): 生成的新的词 ID 序列，形状为 (batch_size, max_new_ids)。
        """
        # 获取输入序列的长度，即初始输入的长度
        input_len = ids.shape[1]

        for t in range(max_new_ids):
            # 显示当前生成的步数（用于调试或监控）
            print(t)

            # 如果序列上下文长度超过模型的 block_size，则截断序列以适应模型的最大输入长度
            if ids.shape[1] <= self.config.block_size:
                # 当前序列作为前文输入
                prev_ids = ids
            else:
                # 截取最后 block_size 个词作为前文输入
                prev_ids = ids[:, -self.config.block_size:]

            # 前向传播：通过模型获取当前前文输入的 logits
            logits = self(prev_ids)  # logits 的形状为 (batch_size, 序列长度, vocab_size)

            # 获取最后一个时间步的 logits，并除以温度以调整采样概率
            logits = logits[:, -1, :] / temperature  # logits 的形状为 (batch_size, vocab_size)

            # 如果指定了 top_k，则对 logits 进行 top-k 截断
            if top_k is not None:
                # 获取当前 logits 中最高的 k 个值及其索引
                v, _ = torch.topk(logits, min(top_k, logits.size(-1))) # v 的形状为 (batch_size, k)
                # 将 logits 中低于第 k 大的值的部分设为负无穷大，以便在 softmax 后概率为0
                logits[logits < v[:, [-1]]] = -float('Inf')

            # 将 logits 转换为概率分布
            probs = F.softmax(logits, dim=-1)  # probs 的形状为 (batch_size, vocab_size)

            # 使用多项式分布进行采样，得到下一个词 ID
            next_id = torch.multinomial(probs, num_samples=1)  # next_id 的形状为 (batch_size, 1)

            # 将采样的下一个词 ID 拼接到当前序列中
            ids = torch.cat((ids, next_id), dim=1)  # ids 的形状更新为 (batch_size, t + 1)

        # 从完整的序列中提取新生成的词 ID
        new_ids = ids[:, input_len:]  # new_ids 的形状为 (batch_size, max_new_ids)

        # 返回生成的新的词 ID 序列
        return new_ids


class Block(nn.Module):
    """
    Transformer 块（Block），包含自注意力机制和前馈神经网络（MLP）。

    每个 Transformer 块由以下部分组成：
    1. 自注意力机制（Self-Attention）
    2. 前馈神经网络（MLP）
    3. 归一化层（Layer Normalization）
    """
    def __init__(self, config: LlamaConfig) -> None:
        """
        初始化 Transformer 块。

        Args:
            config (LlamaConfig): LLaMA 模型的配置参数。
        """
        super().__init__()
        # 自注意力层的归一化层：使用 RMSNorm 进行归一化
        self.att_norm = RMSNorm(config.n_embd) # 输入维度为 n_embd，例如4096

        # 自注意力机制：使用因果自注意力机制（Causal Self-Attention）
        self.att = CausalSelfAttention(config) # 传入配置参数

        # 前馈神经网络层的归一化层：使用 RMSNorm 进行归一化
        self.ffn_norm = RMSNorm(config.n_embd) # 输入维度为 n_embd，例如4096

        # 前馈神经网络：使用 MLP 实现
        self.mlp = MLP(config) # 传入配置参数

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播方法。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, time_steps, hidden_dim)。
            rope (torch.Tensor): 旋转位置嵌入缓存，形状为 (time_steps, head_dim/2)。
            mask (torch.Tensor): 注意力掩码，形状为 (1, 1, time_steps, time_steps)。

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, time_steps, hidden_dim)。
        """
        # 自注意力机制处理：
        # 1. 对输入 x 进行归一化。
        # 2. 应用自注意力机制。
        # 3. 将结果与原始输入 x 相加，实现残差连接。
        x = x + self.att(self.att_norm(x), rope, mask) # x 的形状保持为 (b, t, d)

        # 前馈神经网络处理：
        # 1. 对自注意力机制的输出进行归一化。
        # 2. 应用前馈神经网络。
        # 3. 将结果与自注意力机制的输出相加，实现残差连接。
        x = x + self.mlp(self.ffn_norm(x)) # x 的形状保持为 (b, t, d)

        # 返回处理后的输出张量
        return x


class RMSNorm(nn.Module):
    """
    均方根层归一化（Root Mean Square Layer Normalization，RMSNorm）。

    RMSNorm 是一种归一化方法，与 LayerNorm 类似，但在计算时使用均方根作为缩放因子。
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        初始化 RMSNorm。

        Args:
            dim (int): 输入的维度大小。
            eps (float, 可选): 用于数值稳定的微小值，默认为1e-6。
        """
        super().__init__()
        # 保存微小值
        self.eps = eps
        # 创建一个可学习的缩放因子参数，形状为 (dim,)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        前向传播方法。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, time_steps, dim)。

        Returns:
            torch.Tensor: 归一化后的张量，形状为 (batch_size, time_steps, dim)。
        """
        # 计算输入张量的均方根（Root Mean Square，RMS）
        # 计算每个样本每个时间步的均方值，形状为 (b, t, 1)
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)

        # 计算 RMSNorm 的输出：
        # 1. 对 norm_x 加 eps 以防止除零。
        # 2. 计算均方根（rsqrt 是平方根的倒数）。
        # 3. 将输入张量 x 乘以缩放因子 scale 和均方根。
        output = x * torch.rsqrt(norm_x + self.eps) * self.scale  # 输出形状为 (b, t, d)
        
        # 返回归一化后的张量
        return output


class CausalSelfAttention(nn.Module):
    """
    因果自注意力机制（Causal Self-Attention）类。

    因果自注意力机制确保在预测当前时间步的输出时，模型只能看到过去的时间步，而不能看到未来的时间步。
    这种机制在自回归模型（如语言模型）中非常重要，以保持序列生成的顺序一致性。
    """
    def __init__(self, config: LlamaConfig) -> None:
        """
        初始化因果自注意力机制。

        Args:
            config (LlamaConfig): LLaMA 模型的配置参数。
        """
        super().__init__()
        # 确保嵌入维度可以被头数整除，确保每个头有整数个维度
        assert config.n_embd % config.n_head == 0

        # 为所有头计算键（key）、查询（query）和值（value）的线性投影，但以批处理的方式进行
        # 输入维度为 n_embd，输出维度为 3 * n_embd，因为每个输入会生成三个输出：q, k, v
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)

        # 输出投影线性层：将注意力机制的输出映射回嵌入维度
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

         # 注意力头的数量，例如32
        self.n_head = config.n_head
        # 嵌入维度，例如4096
        self.n_embd = config.n_embd
        # 输入序列的最大长度，例如2048
        self.block_size = config.block_size

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        因果自注意力机制的前向传播方法。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, time_steps, hidden_dim)。
            rope (torch.Tensor): 旋转位置嵌入缓存，形状为 (time_steps, head_dim/2, 2)。
            mask (torch.Tensor): 注意力掩码，形状为 (1, 1, time_steps, time_steps)。

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, time_steps, hidden_dim)。
        """
        # 获取输入张量的批处理大小 (B)、时间步数 (T) 和隐藏维度 (D)
        B, T, D = x.shape

        # 计算查询（q）、键（k）和值（v）
        # self.c_attn(x) 的输出形状为 (B, T, 3*D)
        # 使用 split 分割成三个张量，每个的形状为 (B, T, D)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2) # q, k, v 的形状均为 (B, T, D)
        
        # 重塑键、查询和值的张量形状为 (B, T, n_head, head_dim)
        k = k.view(B, T, self.n_head, D // self.n_head)
        q = q.view(B, T, self.n_head, D // self.n_head)
        v = v.view(B, T, self.n_head, D // self.n_head) 

        # 应用旋转位置嵌入（RoPE）
        # q 的形状保持为 (B, T, n_head, head_dim)
        q = apply_rope(q, rope)
        # k 的形状保持为 (B, T, n_head, head_dim)
        k = apply_rope(k, rope)

        # 转置键、查询和值的张量形状为 (B, n_head, T, head_dim)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # 使用 Flash Attention CUDA 内核进行高效的缩放点积注意力计算
        # attn_mask 用于屏蔽未来的时间步，确保因果关系
        # x 的形状为 (B, n_head, T, head_dim)
        x = F.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            attn_mask=mask, 
            dropout_p=0.0
        )

        # 转置张量形状为 (B, T, n_head, head_dim)，然后重塑为 (B, T, D)
        x = x.transpose(1, 2).contiguous().view(B, T, D) 

        # 应用输出投影线性层，将注意力机制的输出映射回嵌入维度
        # x 的形状保持为 (B, T, D)
        x = self.c_proj(x) 
        
        # 返回输出张量
        return x


class MLP(nn.Module):
    """
    多层感知机（MLP）模块，用于前馈神经网络部分。

    MLP 模块通常位于 Transformer 块的注意力机制之后，用于增加模型的非线性表达能力。
    """
    def __init__(self, config: LlamaConfig) -> None:
        """
        初始化 MLP 模块。

        Args:
            config (LlamaConfig): LLaMA 模型的配置参数。
        """
        super().__init__()

        # 根据配置参数计算隐藏层的维度
        hidden_dim = 4 * config.n_embd  # 隐藏层维度为嵌入维度的4倍，例如4 * 4096 = 16384
        n_hidden = int(2 * hidden_dim / 3)  # 计算第二个隐藏层的维度，为隐藏维度的2/3，例如2 * 16384 / 3 ≈ 10923

        # 第一个线性层：将输入嵌入维度映射到第一个隐藏层维度
        self.c_fc1 = nn.Linear(config.n_embd, n_hidden, bias=False)
        # 第二个线性层：将输入嵌入维度映射到第二个隐藏层维度
        self.c_fc2 = nn.Linear(config.n_embd, n_hidden, bias=False)
        # 输出线性层：将第二个隐藏层维度映射回嵌入维度
        self.c_proj = nn.Linear(n_hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, time_steps, hidden_dim)。

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, time_steps, hidden_dim)。
        """
        # 应用第一个线性层并通过 SiLU 激活函数
        # 将激活后的结果与第二个线性层的输出相乘，实现门控机制
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x) 

        # 应用输出线性层，将结果映射回嵌入维度
        x = self.c_proj(x)  # x 的形状为 (b, t, d)

        # 返回输出张量
        return x


def build_causal_mask(seq_len: int) -> torch.Tensor:
    """
    构建因果掩码（causal mask）。

    因果掩码用于确保在自回归生成过程中，模型在预测当前时间步的输出时，只能看到过去的时间步，而不能看到未来的时间步。

    Args:
        seq_len (int): 序列的长度。

    Returns:
        torch.Tensor: 因果掩码张量，形状为 (1, 1, seq_len, seq_len)。
    """
    # 创建一个全为1的布尔张量，形状为 (seq_len, seq_len)
    ones = torch.ones((seq_len, seq_len), dtype=torch.bool)  # ones 的形状为 (t, t)

    # 使用 torch.tril 获取下三角矩阵，即保留对角线及以下的元素，其余元素设为0
    # 结果 mask 的形状为 (t, t)
    # 在最前面添加两个维度，形状变为 (1, 1, seq_len, seq_len)
    mask = torch.tril(ones)[None, None, :, :]  # mask 的形状为 (1, 1, t, t)

    # 返回因果掩码张量
    return mask
