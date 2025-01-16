import yaml


def parse_yaml(config_yaml: str) -> dict:
    """
    解析 YAML 配置文件。

    参数:
        config_yaml (str): YAML 配置文件的路径。

    返回:
        dict: 解析后的配置字典。
    """
    # 使用 with 语句打开 YAML 配置文件，确保文件会被正确关闭
    with open(config_yaml, "r") as fr:
        # 使用 yaml.FullLoader 加载 YAML 内容，防止潜在的安全问题
        return yaml.load(fr, Loader=yaml.FullLoader)


class LinearWarmUp:
    """
    线性学习率预热调度器。
    
    在训练开始时，学习率会从零逐渐增加到初始学习率，以避免训练初期的不稳定。
    """
    def __init__(self, warm_up_steps: int) -> None:
        """
        初始化线性学习率预热调度器。

        参数:
            warm_up_steps (int): 预热的步数，即在前多少步进行学习率预热。
        """
        # 保存预热的步数
        self.warm_up_steps = warm_up_steps

    def __call__(self, step: int) -> float:
        """
        计算当前步的学习率比例。

        参数:
            step (int): 当前训练的步数。

        返回:
            float: 当前步的学习率比例，范围在 [0, 1] 之间。
                   如果当前步数小于等于预热步数，则返回从0到1的线性增长比例；
                   否则，返回1，表示不再进行预热。
        """
        if step <= self.warm_up_steps:
            # 计算当前步的学习率比例，线性增长
            return step / self.warm_up_steps
        else:
            # 如果当前步数超过预热步数，返回1，表示不再进行预热
            return 1.


def pad_or_truncate(x: list, length: int, pad_value) -> list:
    """
    对列表进行填充或截断，使其达到指定长度。

    参数:
        x (List[Any]): 输入列表。
        length (int): 目标长度。
        pad_value (Any): 用于填充的值。

    返回:
        List[Any]: 填充或截断后的列表。
    """
    if len(x) >= length:
        # 如果列表长度大于或等于目标长度，则截断列表至目标长度
        return x[: length]
    
    else:
        # 如果列表长度小于目标长度，则在末尾填充 pad_value，直到达到目标长度
        return x + [pad_value] * (length - len(x))
