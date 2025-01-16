import random
from typing import Sized


class InfiniteSampler:
    """
    无限采样器：从给定的数据集中无限循环地随机采样索引，不重复使用索引。
    
    Args:
        dataset (Sized): 一个实现了 __len__ 方法的数据集，用于确定数据集的大小。
    """
    def __init__(self, dataset: Sized) -> None:
        """
        初始化无限采样器。

        Args:
            dataset (Sized): 一个实现了 __len__ 方法的数据集，用于确定数据集的大小。
        """
        # 获取数据集的大小，并生成一个包含所有索引的列表
        self.indexes = list(range(len(dataset))) # 例如，如果数据集有5个样本，self.indexes = [0, 1, 2, 3, 4]

        # 对索引列表进行随机打乱，确保采样顺序的随机性
        random.shuffle(self.indexes)  # 例如，shuffle 后 self.indexes = [3, 7, 0, 2, 5, ...]

        # 初始化指针，指向当前采样的位置
        self.p = 0  # 指针初始位置为0，表示从第一个索引开始采样
        
    def __iter__(self):
        """
        使该类成为一个迭代器，返回一个迭代器对象。

        Returns:
            Iterator[int]: 一个整数索引的迭代器。
        """
        # 无限循环，确保可以无限次地采样
        while True:

            if self.p == len(self.indexes):
                """
                当指针达到索引列表的末尾时，表示已经遍历完整个数据集。
                此时需要进行以下操作：
                1. 对索引列表进行重新打乱，以生成新的采样顺序。
                2. 重置指针到列表的开头，准备开始新一轮的采样。
                """
                # 对索引列表进行重新打乱
                random.shuffle(self.indexes) # 例如，shuffle 后 self.indexes = [5, 2, 7, 0, 3, ...]

                # 重置指针到列表的开头
                self.p = 0 # 指针重置为0，准备从第一个索引开始新一轮的采样
            
            # 获取当前指针位置的索引
            index = self.indexes[self.p] # 例如，self.indexes[0] = 3, self.indexes[1] = 7, ...

            # 将指针移动到下一个位置，准备下一次采样
            self.p += 1 # 指针递增，指向下一个索引

            # 返回当前索引，供调用者使用
            yield index 
