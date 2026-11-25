from typing import List


def generate_range(start: float, end: float, step: float) -> List[float]:
    """
    根据步长生成离散数列
    :param start: 开始数值
    :param end: 结束数值
    :param step: 步长
    :return: 离散值数列
    """
    curr = start
    # 获取步长的小数位数，用于 round 精度对齐
    precision = len(str(step).split('.')[-1]) if '.' in str(step) else 0
    while curr <= end + (step / 10):  # 稍微放宽边界判定，防止因极小误差丢掉最大值
        yield round(curr, precision)
        curr += step
