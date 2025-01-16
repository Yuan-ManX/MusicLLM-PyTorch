import dac
import torch
import torch.nn as nn


class DAC(nn.Module):
    """
    DAC（Diffusion Audio Codec）模型，用于音频的编码和解码。
    
    Args:
        sr (float): 采样率，固定为44100 Hz。
        n_quantizers (int): 量化器的数量，用于控制编码的精度。
    """
    def __init__(self, sr: float, n_quantizers: int) -> None:
        """
        初始化 DAC 模型。

        Args:
            sr (float): 采样率，必须为44100 Hz。
            n_quantizers (int): 量化器的数量，用于控制编码的精度。
        """
        super().__init__()

        assert sr == 44100

        # 下载预训练的 DAC 模型，model_type 为 "44khz" 表示44.1kHz的模型
        model_path = dac.utils.download(model_type="44khz")
        # 加载预训练的 DAC 模型
        self.codec = dac.DAC.load(model_path)
        # 保存量化器的数量
        self.n_quantizers = n_quantizers

    def encode(
        self, 
        audio: torch.Tensor, 
    ) -> torch.LongTensor:
        """
        对输入音频进行编码。

        Args:
            audio (torch.Tensor): 输入音频张量，形状为 (batch_size, channels, time_steps)。

        Returns:
            codes (torch.LongTensor): 编码后的代码张量，形状为 (batch_size, quantizers_num, time_steps)。
                                       包含整数类型的代码本索引。
        """
        with torch.no_grad():
            # 设置 codec 为评估模式，禁用梯度计算，节省内存和计算资源
            self.codec.eval()
            # 调用 codec 的 encode 方法进行编码
            # 返回值包括编码后的代码、量化后的特征等
            _, codes, _, _, _ = self.codec.encode(
                audio_data=audio, 
                n_quantizers=self.n_quantizers
            )
            # codes 的形状为 (batch_size, quantizers_num, time_steps)，包含整数类型的代码本索引

        if self.n_quantizers:
            # 如果指定了量化器的数量，则截取相应数量的量化器输出
            codes = codes[:, 0 : self.n_quantizers, :]
            # 调整后的形状仍为 (batch_size, quantizers_num, time_steps)

        # 返回编码后的代码张量
        return codes

    def decode(
        self, 
        codes: torch.LongTensor, 
    ) -> torch.Tensor:
        """
        对编码后的代码进行解码，重建音频信号。

        Args:
            codes (torch.LongTensor): 编码后的代码张量，形状为 (batch_size, quantizers_num, time_steps)。
                                       包含整数类型的代码本索引。

        Returns:
            audio (torch.Tensor): 解码后的音频张量，形状为 (batch_size, channels, time_steps)。
        """
        with torch.no_grad():
            # 设置 codec 为评估模式，禁用梯度计算，节省内存和计算资源
            self.codec.eval()
            # 将代码转换为量化后的特征
            z, _, _ = self.codec.quantizer.from_codes(codes)
            # 调用 codec 的 decode 方法进行解码，重建音频信号
            audio = self.codec.decode(z)

        # 返回解码后的音频张量
        return audio
