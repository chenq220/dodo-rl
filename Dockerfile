FROM dodorl:base

WORKDIR /code

COPY . /code

RUN chmod +x /code/startup.sh \
    /code/scripts/start_distributed.sh \
    /code/scripts/start_torch_rpc_distributed.sh \
    /code/scripts/start_compose_distributed.sh \
    /code/scripts/compose_role.sh

ENTRYPOINT ["/code/startup.sh"]
