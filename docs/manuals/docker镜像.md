# Docker 镜像

镜像分两层构建，不能直接 `docker build .`：

1. `Dockerfile-base` 构建 `dodorl:base`，只包含 Python 依赖和系统工具，不含业务代码。
2. `Dockerfile` 基于 `dodorl:base` 复制本仓库代码，生成可运行镜像。

```bash
# 1) 基础镜像：无业务代码，分布式 compose 直接使用它
./scripts/docker-build.sh base

# 2) 带代码镜像：按需打 dev 或版本标签
./scripts/docker-build.sh v0.17.0
```

不使用脚本时，等价命令为：

```bash
docker build --platform linux/amd64 -f Dockerfile-base -t dodorl:base .
docker build --platform linux/amd64 -f Dockerfile -t dodorl:v0.17.0 .
```

`scripts/docker-compose-distribute.yml` 默认使用 `dodorl:base`，并把仓库目录 bind 挂载到容器 `/code`，因此运行分布式训练只需先执行 `./scripts/docker-build.sh base`。首次构建会拉取 `python:3.12.13-slim-trixie` 并安装 `requirements.txt`，耗时较长；只有在依赖或系统工具变化时才需要重建基础镜像。
