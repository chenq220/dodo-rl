import os
from kubernetes import client, config
from common.logger import init_logger

LOG = init_logger('k8s_engine')


# 要执行本操作需要在k8s中给pod打标签的权限，导入./pod_role.yaml
def mark_pod_tag(tag_kv):
    if not in_k8s_env():
        return
    # 1. 从环境变量获取自己的身份
    pod_name = os.getenv("MY_POD_NAME")
    namespace = os.getenv("MY_POD_NAMESPACE", "default")

    if not pod_name:
        LOG.error("Not detected Pod name，Probably not in k8s.")
        return

    # 2. 加载集群内认证配置
    config.load_incluster_config()
    v1 = client.CoreV1Api()

    # 3. 准备标签数据
    body = {
        "metadata": {
            "labels": tag_kv
        }
    }

    # 4. 调用 API 给自己打标签 (Patch 操作)
    try:
        v1.patch_namespaced_pod(name=pod_name, namespace=namespace, body=body)
    except Exception as e:
        LOG.error(f"Failed to tag: {e}")


def in_k8s_env():
    return True if os.environ.get('KUBERNETES_SERVICE_HOST') else False


def get_sts_info():
    # 1. 加载配置
    config.load_incluster_config()
    apps_v1 = client.AppsV1Api()

    # 2. 获取 Pod 名字并解析出 StatefulSet 名字
    pod_name = os.getenv("MY_POD_NAME")
    sts_name = "-".join(pod_name.split("-")[:-1])
    namespace = os.getenv("MY_POD_NAMESPACE", "default")

    # 3. 查询 STS 属性
    sts = apps_v1.read_namespaced_stateful_set(name=sts_name, namespace=namespace)
    return sts


if __name__ == '__main__':
    mark_pod_tag({"task-status": "finished"})
