import os
import time
from functools import wraps
from common.logger import init_logger

LOG = init_logger('rollout_worker')


def file_lock_decorator(lock_name):
    """
    分布式目录锁装饰器
    :param lock_name: 锁文件夹的名字，如 'state_json.lock'
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, state_raw, json_file, *args, **kwargs):
            # 1. 确定锁的完整路径（放在与目标文件相同的目录）
            lock_path = os.path.join(os.path.dirname(json_file), lock_name)

            retry_interval = 0.1  # 重试间隔
            max_timeout = 10  # 最大等待时间
            start_time = time.time()
            acquired = False

            # 2. 尝试获取锁 (自旋锁)
            while time.time() - start_time < max_timeout:
                try:
                    # os.mkdir 是原子操作，跨主机共享磁盘也适用
                    os.mkdir(lock_path)
                    acquired = True
                    break
                except FileExistsError:
                    # 锁已被占用，等待后重试
                    time.sleep(retry_interval)
                except Exception as e:
                    LOG.error(f"创建分布式锁异常: {e}")
                    raise

            if not acquired:
                raise TimeoutError(f"获取分布式锁 {lock_name} 超时，请检查是否有进程死锁")

            # 3. 执行业务逻辑
            try:
                # 注意：为了保证 JSON 完整性，这里内部依然建议使用 r+ 或 w
                return func(self, state_raw, json_file, *args, **kwargs)
            finally:
                # 4. 释放锁：删除文件夹
                try:
                    os.rmdir(lock_path)
                except Exception as e:
                    LOG.error(f"释放分布式锁失败: {e}")

        return wrapper

    return decorator
