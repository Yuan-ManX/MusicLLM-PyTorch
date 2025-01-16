import re
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoTokenizer

from utils import pad_or_truncate


class BertDacTokenizer:
    """
    扩展的文本分词器，集成了离散音频编解码器（Discrete Audio Codec）的词汇表。
    """
    
    def __init__(self, audio_codec: nn.Module) -> None:
        """
        初始化 BertDacTokenizer。

        Args:
            audio_codec (nn.Module): 音频编解码器模型，包含音频编解码器的词汇表信息。
        """
        super().__init__()
        # 初始化 BERT 分词器，使用预训练的 "bert-base-uncased" 模型
        self.tok = AutoTokenizer.from_pretrained("bert-base-uncased")

        # 获取音频编解码器的属性
        self.codebook_size = audio_codec.codec.codebook_size  # 获取代码本大小，例如1024
        self.n_quantizers = audio_codec.n_quantizers  # 获取量化器的数量，例如2

        # 定义音频开始（boa）和音频结束（eoa）的特殊标记
        new_vocabs = ["<boa>", "<eoa>"]

        # 生成音频编解码器的词汇表
        for q in range(self.n_quantizers):
            for i in range(self.codebook_size):
                # 为每个量化器和代码本索引生成一个唯一的标记名称，例如 "dac_l0_0", "dac_l0_1", ..., "dac_l1_1023"
                new_vocabs.append("dac_l{}_{}".format(q, i))

        # 输出原始词汇表大小
        print("Original vocab size: {}".format(len(self.tok)))
        # 输出音频词汇表大小
        print("Audio vocab size: {}".format(len(new_vocabs)))
        # 将音频词汇表添加到 BERT 分词器中
        self.tok.add_tokens(new_vocabs)
        # 输出扩展后的词汇表大小
        print("Extended vocab size: {}".format(len(self.tok)))

    def captions_to_ids(
        self,
        captions: list[str], 
        fix_length: int
    ) -> torch.LongTensor:
        """
        将文本标题转换为对应的 ID 序列。

        例如，["Hello world", "rock"]
           -> [[101, 8667, 1362, 102, 0, 0], [101, 2067, 102, 0, 0, 0]]

        Args:
            captions (List[str]): 输入的文本标题列表。
            fix_length (int): 固定的长度，用于填充或截断。

        Returns:
            batch_ids (torch.LongTensor): 转换后的 ID 序列张量，形状为 (batch_size, fix_length)。
        """

        batch_ids = []
        for caption in captions:

            # 将文本标题转换为 tokens
            tokens = self.tok.tokenize(caption)

            # 将 tokens 转换为 IDs，并保留前 (fix_length - 2) 个 IDs，保留两个位置用于特殊标记
            ids = self.tok.convert_tokens_to_ids(tokens)[0 : fix_length - 2]

            # 在开头添加 [CLS] 标记，在末尾添加 [SEP] 标记
            ids = [self.tok.cls_token_id] + ids + [self.tok.sep_token_id]

            # 如果需要固定长度，则进行填充或截断
            if fix_length:
                ids = pad_or_truncate(ids, fix_length, self.tok.pad_token_id)

            # 将处理后的 IDs 添加到批处理列表中
            batch_ids.append(ids)

        # 将批处理列表转换为 PyTorch LongTensor
        return torch.LongTensor(batch_ids)

    def audio_codes_to_ids(self, codes: torch.LongTensor) -> torch.LongTensor:
        """
        将音频代码转换为对应的 tokens，然后转换为 IDs。

        例如：
            audio_codes: [[[568, 568, 568], [778, 778, 804]]] 

         -> tokens: [["<boa>", "dac_l0_568", "dac_l1_778", "dac_l0_568", 
                      "dac_l1_778", "dac_l0_568", "dac_l1_804", "<eoa>"]]
        
         -> IDs: [[30522, 31092, 32326, 31092, 32326, 31092, 32352, 30523]]

        Args:
            codes (torch.LongTensor): 输入的音频代码张量，形状为 (batch_size, quantizers_num, time_steps)。

        Returns:
            batch_ids (torch.LongTensor): 转换后的 ID 序列张量，形状为 (batch_size, quantizers_num * time_steps + 2)。
        """
        # 获取设备信息（CPU 或 GPU）
        device = codes.device
        # 获取批处理大小 (B)、量化器数量 (Q) 和时间步数 (T)
        B, Q, T = codes.shape

        # 将音频代码张量从 GPU 移动到 CPU，并转换为 NumPy 数组
        codes = codes.cpu().numpy()
        # 初始化一个与 codes 形状相同的 NumPy 数组，用于存储转换后的 IDs
        batch_ids = np.zeros_like(codes, dtype="int64")

        # 遍历每个批次、每个量化器和每个时间步，将音频代码转换为对应的 token，然后转换为 ID
        for b in range(B):
            for q in range(Q):
                for t in range(T):
                    token = "dac_l{}_{}".format(q, codes[b, q, t]) # 生成 token，例如 "dac_l0_568"
                    batch_ids[b, q, t] = self.tok.convert_tokens_to_ids(token) # 转换为 ID

        # 使用 einops 库将 (B, Q, T) 重塑为 (B, Q*T)
        batch_ids = rearrange(batch_ids, 'b q t -> b (t q)')

        # Special tokens
        # 生成音频开始的 ID 列表，形状为 (B, 1)
        boa_ids = np.ones((B, 1), dtype="int64") * self.tok.convert_tokens_to_ids("<boa>")
        # 生成音频结束的 ID 列表，形状为 (B, 1)
        eoa_ids = np.ones((B, 1), dtype="int64") * self.tok.convert_tokens_to_ids("<eoa>")

        # 将音频开始、音频代码和音频结束连接起来，形状为 (B, Q*T + 2)
        batch_ids = np.concatenate((boa_ids, batch_ids, eoa_ids), axis=-1)  # shape: (b, t)
        # 将 NumPy 数组转换为 PyTorch LongTensor，并移动回原始设备
        batch_ids = torch.LongTensor(batch_ids).to(device)

        # 返回转换后的 ID 序列张量
        return batch_ids

    def ids_to_audio_codes(self, ids: torch.LongTensor) -> torch.LongTensor:
        """
        将 IDs 转换为音频 tokens，然后转换为音频代码。

        例如：
            IDs: [[30522, 31092, 32326, 31092, 32326, 31092, 32352, 30523]]

         -> tokens: [["<boa>", "dac_l0_568", "dac_l1_778", "dac_l0_568", 
                      "dac_l1_778", "dac_l0_568", "dac_l1_804", "<eoa>"]]
        
         -> audio_codes: [[[568, 568, 568], [778, 778, 804]]]

        Args:
            codes (torch.LongTensor): 输入的 ID 序列张量，形状为 (batch_size, quantizers_num * time_steps + 2)。

        Returns:
            batch_codes (torch.LongTensor): 转换后的音频代码张量，形状为 (batch_size, quantizers_num, time_steps)。
        """
        # 获取设备信息（CPU 或 GPU）
        device = ids.device
        # 获取批处理大小 (B) 和序列长度 (T)
        B, T = ids.shape

        # 将 ID 张量从 GPU 移动到 CPU，并转换为 NumPy 数组
        ids = ids.cpu().numpy()
        # 初始化一个空列表，用于存储每个批次的音频代码
        batch_codes = []

        # 遍历每个批次的 ID 序列
        for b in range(B):
            # 将 IDs 转换为 tokens
            tokens = self.tok.convert_ids_to_tokens(ids[b])
            # 初始化一个空列表，用于存储当前批次的音频代码
            codes = []

            for t in range(T):
                # 获取当前 token
                token = tokens[t]
                # 使用正则表达式匹配 token 格式
                match = re.match(r'dac_l(\d+)_(\d+)', token)

                if match:
                    # 获取量化器索引
                    q = int(match.groups()[0])
                    # 获取代码本索引
                    id = int(match.groups()[1])

                    if q == 0:
                        # 如果是第一个量化器，则初始化缓冲区
                        buffer = []

                    # 将代码本索引添加到缓冲区
                    buffer.append(id)

                    if q == self.n_quantizers - 1:
                        if len(buffer) == self.n_quantizers:
                            # 如果缓冲区包含所有量化器的代码，则添加到 codes 列表中
                            codes.append(buffer)
                            # 重置缓冲区
                            buffer = []

            # 将 codes 列表转换为 PyTorch LongTensor，并调整形状为 (quantizers_num, time_steps)
            codes = torch.LongTensor(codes)
            # 重塑形状为 (quantizers_num, time_steps)
            codes = rearrange(codes, 't q -> q t')  # shape: (q, t)
            # 将当前批次的音频代码添加到 batch_codes 列表中
            batch_codes.append(codes)

        # 将 batch_codes 列表转换为 PyTorch 张量，并移动回原始设备
        batch_codes = torch.stack(batch_codes, dim=0).to(device)  # 形状为 (batch_size, quantizers_num, time_steps)

        # 返回转换后的音频代码张量
        return batch_codes

    def __len__(self):
        """
        返回词汇表的大小。

        Returns:
            int: 词汇表的大小。
        """
        return len(self.tok)

    @property
    def pad_token_id(self):
        """
        获取填充符的 ID。

        Returns:
            int: 填充符的 ID。
        """
        return self.tok.pad_token_id

    @property
    def boa_token_id(self):
        """
        获取音频开始标记的 ID。

        Returns:
            int: 音频开始标记的 ID。
        """
        return self.tok.convert_tokens_to_ids("<boa>")

    @property
    def eoa_token_id(self):
        """
        获取音频结束标记的 ID。

        Returns:
            int: 音频结束标记的 ID。
        """
        return self.tok.convert_tokens_to_ids("<eoa>")
