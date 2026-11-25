def encode_discrete_action(action, action_space_sizes):
    """
    将离散动作数组编码为单个整数

    参数:
        action: 动作数组，如 [0, 1]（二维）或 [2]（一维）
        action_space_sizes: 每个动作维度的离散大小，如 [3, 4] 表示第一维有3个动作，第二维有4个动作

    返回:
        编码后的整数
    """
    # 确保动作维度与空间大小匹配
    if len(action) != len(action_space_sizes):
        raise ValueError(
            f"动作维度 {len(action)} 与空间维度 {len(action_space_sizes)} 不匹配"
        )

    # 检查每个动作是否在有效范围内
    for a, size in zip(action, action_space_sizes):
        if not (0 <= a < size):
            raise ValueError(f"动作 {a} 超出范围 [0, {size - 1}]")

    # 计算编码值（类似多维数组展平）
    encoded = 0
    for i, (a, size) in enumerate(zip(action, action_space_sizes)):
        # 计算当前维度之前的总空间大小（乘积）
        prior_size = 1
        for j in range(i + 1, len(action_space_sizes)):
            prior_size *= action_space_sizes[j]
        encoded += a * prior_size

    return encoded


def decode_discrete_action(encoded_action, action_space_sizes):
    """解码：将整数还原为动作数组"""
    action = []
    remaining = encoded_action
    for i in range(len(action_space_sizes)):
        # 计算当前维度的除数
        divisor = 1
        for j in range(i + 1, len(action_space_sizes)):
            divisor *= action_space_sizes[j]

        # 计算当前维度的动作值
        a = remaining // divisor
        action.append(a)
        remaining = remaining % divisor

    return action
