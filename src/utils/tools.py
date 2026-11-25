import psutil
import random
import socket
import ipaddress
import toml
import numpy as np
import gymnasium as gym


def get_local_ipaddr(nic_name=None):
    """
    获取本机 IP 地址。

    Args:
        nic_name (str, optional): 指定网卡名称（如 'eth0', 'en0', 'WLAN'）。
                                 如果为 None，则返回第一个非回环网卡的第一个 IPv4 地址。

    Returns:
        str: IP 地址字符串。如果未找到则返回 None。
    """
    addrs = psutil.net_if_addrs()

    if nic_name:
        if nic_name in addrs:
            for addr in addrs[nic_name]:
                # 过滤 IPv4 地址 (socket.AF_INET)
                if addr.family == socket.AF_INET:
                    return addr.address
            return None  # 该网卡没有 IPv4
        else:
            raise ValueError(f"未找到网卡: {nic_name}")

    for name, snics in addrs.items():
        # 排除回环接口 (Loopback)
        # 在 Linux/macOS 通常叫 'lo'，Windows 通常包含 'Loopback'
        if name.lower().startswith('lo') or 'loopback' in name.lower():
            continue

        for snic in snics:
            if snic.family == socket.AF_INET:
                # 额外校验：确保不是 127.0.0.1
                if not ipaddress.ip_address(snic.address).is_loopback and ipaddress.ip_address(
                        snic.address).version == 4:
                    return snic.address

    return None


def load_config(config_file_path: str) -> dict:
    with open(config_file_path, "r", encoding="utf-8") as f:
        print(f"config_file path is {config_file_path}")
        return toml.load(f)


def zeros_ndarray(space: gym.spaces.Space) -> np.ndarray:
    """
    返回指定智能体的全0观测向量
    处理不同观测空间类型（Box/Dict/Tuple），避免 shape 访问错误
    """
    if isinstance(space, gym.spaces.Box):
        return np.zeros(shape=space.shape, dtype=space.dtype)
    elif isinstance(space, gym.spaces.Discrete):
        return np.array([0], dtype=np.int32)
    raise NotImplementedError(f"不支持的观测空间类型 {type(space)}")


def get_available_udp_port_fast(target_ip='0.0.0.0', port_range=(50000, 60000)):
    """
    通过集合差集快速获取指定 IP 上未被占用的 UDP 端口
    """
    full_pool = set(range(port_range[0], port_range[1] + 1))
    used_ports = set()
    try:
        connections = psutil.net_connections(kind='udp')
    except psutil.AccessDenied:
        connections = []

    # 获取本机在用的udi端口号
    for conn in connections:
        # 如果连接没有绑定地址，跳过
        if not conn.laddr:
            continue

        # 检查逻辑：
        # 如果该端口绑定在 target_ip 上，或者绑定在 0.0.0.0 / :: (所有网卡)
        if conn.laddr.ip == target_ip or conn.laddr.ip in ('0.0.0.0', '::'):
            used_ports.add(conn.laddr.port)

    # 计算差集：在全集中但不在占用集中的端口
    available_pool = list(full_pool - used_ports)

    if not available_pool:
        raise RuntimeError(f"No available udp port in range{port_range}")

    # 4. 从可用池中随机抽取一个
    return random.choice(available_pool)

