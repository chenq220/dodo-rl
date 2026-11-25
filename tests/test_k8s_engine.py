import pytest


from engine.k8s_engine import in_k8s_env, mark_pod_tag


def test_in_k8s_env_false_by_default(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    assert in_k8s_env() is False


def test_in_k8s_env_true_when_host_set(monkeypatch):
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    assert in_k8s_env() is True


def test_mark_pod_tag_noop_outside_cluster(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    mark_pod_tag({"task-status": "finished"})


def test_mark_pod_tag_patches_pod(monkeypatch):
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    monkeypatch.setenv("MY_POD_NAME", "dodo-0")
    monkeypatch.setenv("MY_POD_NAMESPACE", "rl")
    patched = {}

    class FakeApi:
        def patch_namespaced_pod(self, name, namespace, body):
            patched["name"] = name
            patched["namespace"] = namespace
            patched["body"] = body

    monkeypatch.setattr("engine.k8s_engine.config.load_incluster_config", lambda: None)
    monkeypatch.setattr("engine.k8s_engine.client.CoreV1Api", FakeApi)
    mark_pod_tag({"task-status": "finished"})
    assert patched["name"] == "dodo-0"
    assert patched["namespace"] == "rl"
    assert patched["body"]["metadata"]["labels"]["task-status"] == "finished"
