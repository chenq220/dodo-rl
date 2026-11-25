from typing import Any
from common.logger import init_logger

LOG = init_logger('exception')


class DodoException(Exception):
    """Dodo 框架全局异常基类"""

    msg = "An unknown exception occurred."

    def __init__(self, **kwargs):
        message = getattr(self, 'msg', DodoException.msg)

        if kwargs:
            try:
                message = message % kwargs
            except Exception as e:
                LOG.error(f"Failed to format exception message: {e}")

        self.msg = message

        super().__init__(self.msg)


class VarNotDefinedException(DodoException):
    msg = "Variable %(name)s is not defined."


class ValueEumException(DodoException):
    msg = "Parameter values do not meet the enum criteria"


class KeyErrorException(DodoException):
    """字典或映射中缺少指定键。"""

    msg = "Key %(key)s not found in %(container)s."

    def __init__(self, key: Any = "", container: str = ""):
        DodoException.__init__(self, key=key, container=container)


class ValueErrorException(DodoException):
    """变量取值不符合要求。"""

    msg = "Invalid value for %(name)s. Expected: %(expected)s, Given: %(given)s."

    def __init__(self, name: str = "", expected: Any = "", given: Any = ""):
        DodoException.__init__(self, name=name, expected=expected, given=given)


class TaskIdFrozenException(DodoException):
    """当前进程任务身份冻结后尝试切换到另一任务。"""

    msg = (
        "Task ID is frozen. Current: %(current_task_id)s, "
        "Configured: %(configured_task_id)s"
    )

    def __init__(
            self,
            current_task_id: str = "",
            configured_task_id: str = "",
    ):
        DodoException.__init__(
            self,
            current_task_id=current_task_id,
            configured_task_id=configured_task_id,
        )


class DataInvalidException(DodoException):
    msg = "Data %(name)s is invalid or not excited."


class PluginNameDuplicateException(DodoException):
    """同一插件类别下的名称被重复定义。"""

    msg = (
        "Plugin name is duplicated. Type: %(plugin_type)s, Name: %(name)s, "
        "Existing: %(existing)s, Incoming: %(incoming)s"
    )

    def __init__(
            self,
            plugin_type: str = "",
            name: str = "",
            existing: Any = "",
            incoming: Any = "",
    ):
        DodoException.__init__(
            self,
            plugin_type=plugin_type,
            name=name,
            existing=existing,
            incoming=incoming,
        )


class PluginImplementationDuplicateException(DodoException):
    """多个 module 插件尝试替换同一个 @extensible 基线类。"""

    msg = (
        "Plugin implementation is duplicated. Baseline: %(baseline)s, "
        "Existing plugin: %(existing)s, Incoming plugin: %(incoming)s"
    )

    def __init__(self, baseline: str = "", existing: str = "", incoming: str = ""):
        DodoException.__init__(
            self,
            baseline=baseline,
            existing=existing,
            incoming=incoming,
        )


class SampleTaskReturned(DodoException):
    """
    采样任务因仿真异常被归还任务总线。

    该异常不是业务失败，也不是训练结束信号；它表示当前 episode 没有产出有效样本，
    Rollout 已经把采样令牌还给 TrainTaskManager，角色层应跳过本轮后续学习并重新进入调度循环。
    """
    msg = "Sample task has been returned to task manager. Episode: %(episode_num)s"

    def __init__(self, episode_num: Any = ""):
        DodoException.__init__(self, episode_num=episode_num)


class SampleLeaseException(DodoException):
    """
    采样租约协议违规。

    覆盖三类不可自愈的簿记错误：回报缺租约、回显身份与租约表登记不符、
    以及「无在途、无可领、却未完成」的进度死角。预期内的迟到回报
    （超时回收、换代清场后到达）不属于本异常，由丢弃计数观测。
    """
    msg = "Sample lease protocol error. Operation: %(operation)s, Detail: %(detail)s"

    def __init__(self, operation: str = "", detail: Any = ""):
        DodoException.__init__(self, operation=operation, detail=detail)


class ModelVersionNotFound(ValueError):
    """An explicit model snapshot has not been registered."""


class ModelDimException(DodoException):
    """
    模型维度不匹配。
    """
    msg = "Model dimension %(dim_name)s does not match. Expected: %(expected)s, Given: %(given)s"


class ModelCheckpointCompatibilityException(DodoException):
    """checkpoint 缺少权重、张量形状或可验证的网络维度与当前架构不兼容。"""

    msg = "Model checkpoint incompatible. Agent: %(agent_name)s, Model: %(model_name)s, Detail: %(detail)s"

    def __init__(self, agent_name: str = "", model_name: str = "", detail: Any = ""):
        DodoException.__init__(self, agent_name=agent_name, model_name=model_name, detail=detail)


class EnvAlgoDimException(DodoException):
    """环境全局状态维度与算法配置 state_shape 不一致。"""

    msg = (
        "Environment and algorithm dimensions do not match. "
        "Config: %(config_name)s, EnvConfig.state_dim: %(expected)s, state_shape: %(given)s"
    )

    def __init__(self, config_name: str = "", expected: Any = "", given: Any = ""):
        DodoException.__init__(self, config_name=config_name, expected=expected, given=given)


class ReplayBufferNoEnoughException(DodoException):
    """
    缓存池不足
    """
    msg = "Replay buffer has no enough enough datas."


class S3ProtocolException(DodoException):
    """S3 协议存储失败（建桶、上传、下载等）。"""

    msg = (
        "S3 protocol error. Operation: %(operation)s, Bucket: %(bucket)s, "
        "Key: %(key)s, Detail: %(detail)s"
    )

    def __init__(
            self,
            operation: str = "",
            bucket: str = "",
            key: str = "",
            detail: Any = "",
    ):
        self.operation = operation
        self.bucket = bucket
        self.key = key
        self.detail = detail
        DodoException.__init__(
            self,
            operation=operation,
            bucket=bucket,
            key=key,
            detail=detail,
        )


class EpisodeInterruptedException(DodoException):
    """当前 episode 无法产生有效样本，调用方应归还采样令牌。"""

    msg = "Episode interrupted. Detail: %(detail)s"

    def __init__(self, detail: Any = ""):
        DodoException.__init__(self, detail=detail)


class RpcException(DodoException):
    """Dodo RPC 异常基类。"""

    msg = "RPC error. Service: %(service)s, Method: %(method)s, Detail: %(detail)s"

    def __init__(self, service: str = "", method: str = "", detail: Any = ""):
        self.service = service
        self.method = method
        self.detail = detail
        DodoException.__init__(self, service=service, method=method, detail=detail)


class RecoverableRpcException(EpisodeInterruptedException, RpcException):
    """会使当前 episode 作废、但允许归还令牌后重试的 RPC 异常。"""

    msg = "Recoverable RPC error. Service: %(service)s, Method: %(method)s, Detail: %(detail)s"

    def __init__(self, service: str = "", method: str = "", detail: Any = ""):
        self.service = service
        self.method = method
        self.detail = detail
        DodoException.__init__(self, service=service, method=method, detail=detail)


class RpcCodecException(RecoverableRpcException):
    """RPC 请求或响应的 pickle 编解码失败。"""


class RpcTransportException(RecoverableRpcException):
    """RPC 网络、超时或资源层面的暂时失败。"""

    def __init__(self, service: str = "", method: str = "", detail: Any = "", status_code: Any = ""):
        self.status_code = status_code
        super().__init__(service=service, method=method, detail=detail)


class RpcRemoteApplicationException(RpcException):
    """远端业务方法执行失败；默认不视为可恢复链路故障。"""

    def __init__(self, service: str = "", method: str = "", detail: Any = "",
                 remote_error_type: str = "", remote_traceback: str = ""):
        self.remote_error_type = remote_error_type
        self.remote_traceback = remote_traceback
        super().__init__(service=service, method=method, detail=detail)


class RpcServerGone(RpcException):
    """TrainTaskManager 服务结束；消费角色据此退出，不归还 episode。"""

    def __init__(self, method: str, attempts: int, interval: float, detail: str = "",
                 service: str = "TrainTaskManager"):
        self.attempts = attempts
        self.interval = interval
        super().__init__(
            service=service,
            method=method,
            detail=f"server unreachable after {attempts} attempts (interval={interval}s): {detail}",
        )


class RpcWorldFailure(RpcException):
    """当前 TensorPipe agent/pipe 已失效；同一静态 RPC world 内不可原地恢复。"""

    msg = (
        "Torch RPC world failure. Task: %(task_id)s, Worker: %(worker)s, Rank: %(rank)s, "
        "Sample role: %(sample_role)s, Service: %(service)s, Method: %(method)s, "
        "Episode: %(episode)s, Generation: %(generation)s, Detail: %(detail)s"
    )

    def __init__(
            self,
            *,
            task_id: str = "",
            worker: str = "",
            rank: Any = "",
            sample_role: str = "",
            service: str = "",
            method: str = "",
            episode: Any = "",
            generation: Any = "",
            detail: Any = "",
    ):
        self.task_id = task_id
        self.worker = worker
        self.rank = rank
        self.sample_role = sample_role
        self.episode = episode
        self.generation = generation
        self.original_detail = str(detail)
        self.service = service
        self.method = method
        self.detail = detail
        DodoException.__init__(
            self,
            task_id=task_id,
            worker=worker,
            rank=rank,
            sample_role=sample_role,
            service=service,
            method=method,
            episode=episode,
            generation=generation,
            detail=detail,
        )


class SimulationException(EpisodeInterruptedException):
    """
    仿真交互异常基类。

    Rollout只捕获这类异常并整局重试
    """
    msg = "Simulation exception. Detail: %(detail)s"

    def __init__(self, detail: Any = ""):
        DodoException.__init__(self, detail=detail)


class SimulationTimeoutException(SimulationException):
    """
    仿真链路超时。
    """
    msg = "Simulation timeout. Stage: %(stage)s, Endpoint: %(endpoint)s, Timeout: %(timeout)s, Detail: %(detail)s"

    def __init__(self, stage: str, endpoint: str = "", timeout: Any = "", detail: Any = ""):
        DodoException.__init__(self, stage=stage, endpoint=endpoint, timeout=timeout, detail=detail)


class SimulationStateNotReceivedException(SimulationException):
    """
    未等到满足条件的仿真态势数据。
    """
    msg = "Simulation state data not received. Time point: %(time_point)s, Timeout: %(timeout)s, Detail: %(detail)s"

    def __init__(self, time_point: Any = "", timeout: Any = "", detail: Any = ""):
        DodoException.__init__(self, time_point=time_point, timeout=timeout, detail=detail)


class MiddlewareException(SimulationException):
    """中间件建连、bind、发送等传输失败。"""

    msg = "Middleware error. Stage: %(stage)s, Endpoint: %(endpoint)s, Detail: %(detail)s"

    def __init__(self, stage: str, endpoint: str = "", detail: Any = ""):
        DodoException.__init__(self, stage=stage, endpoint=endpoint, detail=detail)


class ZMQMiddlewareException(SimulationException):
    """
    ZMQ中间件异常。
    """
    msg = "ZMQ middleware error. Stage: %(stage)s, Endpoint: %(endpoint)s, Detail: %(detail)s"

    def __init__(self, stage: str, endpoint: str = "", detail: Any = ""):
        DodoException.__init__(self, stage=stage, endpoint=endpoint, detail=detail)


class SimulationCommandException(SimulationException):
    """
    仿真指令执行失败。
    """
    msg = "Simulation command error. Command: %(command)s, Reply: %(reply)s, Detail: %(detail)s"

    def __init__(self, command: Any, reply: Any = "", detail: Any = ""):
        DodoException.__init__(self, command=command, reply=reply, detail=detail)


class EnvironmentException(EpisodeInterruptedException):
    """
    环境建模与环境业务逻辑异常基类。

    在线采样期间抛出该异常表示当前 episode 的环境样本无效；
    Rollout 会归还采样令牌，由后续采样流程整局重试。
    """
    msg = "Environment exception. Detail: %(detail)s"

    def __init__(self, detail: Any = ""):
        DodoException.__init__(self, detail=detail)


class AgentRoutingException(EnvironmentException):
    """Rollout Agent编排与策略路由异常基类。"""

    msg = "Agent routing failed. Detail: %(detail)s"

    def __init__(self, detail: Any = ""):
        DodoException.__init__(self, detail=detail)


class AgentOwnerDuplicateException(AgentRoutingException):
    """同一个环境AgentID被多个Step Agent声明管理。"""

    msg = "Agent owner is duplicated. AgentID: %(agent_id)s, Owners: %(owners)s"

    def __init__(self, agent_id: Any = "", owners: Any = ""):
        DodoException.__init__(self, agent_id=agent_id, owners=owners)


class AgentOwnerNotFoundException(AgentRoutingException):
    """顺序规则返回的环境AgentID没有对应的Step Agent。"""

    msg = "Agent owner not found. AgentID: %(agent_id)s, Owned AgentIDs: %(owned_agent_ids)s"

    def __init__(self, agent_id: Any = "", owned_agent_ids: Any = ""):
        DodoException.__init__(self, agent_id=agent_id, owned_agent_ids=owned_agent_ids)


class AgentSelectionNotFoundException(AgentRoutingException):
    """顺序规则返回的Agent不在当前待路由表中。"""

    msg = (
        "Selected agent was not found in routing table. Scope: %(scope)s, "
        "Selected: %(selected)s, Available: %(available)s"
    )

    def __init__(self, scope: str = "", selected: Any = "", available: Any = ""):
        DodoException.__init__(
            self,
            scope=scope,
            selected=selected,
            available=available,
        )


class AgentActionCoverageException(AgentRoutingException):
    """Parallel步进前生成的动作未覆盖全部活动环境Agent。"""

    msg = "Agent actions do not cover active agents. Expected: %(expected)s, Actual: %(actual)s"

    def __init__(self, expected: Any = "", actual: Any = ""):
        DodoException.__init__(self, expected=expected, actual=actual)


class AgentGroupConfigException(DodoException):
    """Agent 无名分组配置非法（入口冲突、空组、组宽不一致或跨组重复成员）。"""

    msg = "Agent group config invalid. Agent: %(agent_name)s, Detail: %(detail)s"

    def __init__(self, agent_name: Any = "", detail: Any = ""):
        DodoException.__init__(self, agent_name=agent_name, detail=detail)


class AgentGroupCompatibilityException(DodoException):
    """显式分组与环境/Agent/Learner 类型或维度契约不兼容。"""

    msg = (
        "Explicit agent grouping is incompatible with the current assembly. "
        "Agent: %(agent_name)s, Detail: %(detail)s"
    )

    def __init__(self, agent_name: Any = "", detail: Any = ""):
        DodoException.__init__(self, agent_name=agent_name, detail=detail)


class ReplayBatchShapeException(DodoException):
    """入池或组批中同一 AgentName 的轨迹类型/组宽/成员顺序不一致。"""

    msg = "Replay batch transition layout mismatch. Agent: %(agent_name)s, Detail: %(detail)s"

    def __init__(self, agent_name: Any = "", detail: Any = ""):
        DodoException.__init__(self, agent_name=agent_name, detail=detail)


class ReplayPackagingException(DodoException):
    """样本封装/抽样模式与配置或策略不兼容。"""

    msg = "Replay sample packaging error. Detail: %(detail)s"

    def __init__(self, detail: Any = ""):
        DodoException.__init__(self, detail=detail)


class AgentComposeException(DodoException):
    """AgentCompose配置无法驱动当前Rollout流程。"""

    msg = "Agent compose error. Compose type: %(compose_type)s, Detail: %(detail)s"

    def __init__(self, compose_type: Any = "", detail: Any = ""):
        DodoException.__init__(self, compose_type=compose_type, detail=detail)


class EnvironmentTypeException(EnvironmentException):
    """Rollout入口接收到不支持或不匹配的环境基类。"""

    msg = (
        "Environment type error. Operation: %(operation)s, "
        "Expected: %(expected)s, Given: %(given)s"
    )

    def __init__(self, operation: str = "", expected: Any = "", given: Any = ""):
        DodoException.__init__(
            self,
            operation=operation,
            expected=expected,
            given=given,
        )


class StateEntityNotFoundException(EnvironmentException):
    """
    态势中检索不到指定实体。
    """
    msg = "State entity not found. Name: %(name)s, Available: %(available)s"

    def __init__(self, name: Any, available: Any = ""):
        DodoException.__init__(self, name=name, available=available)


class CoordinateFrameException(EnvironmentException):
    """态势坐标系与 Frame 类型不一致，或转换原点类型不匹配。"""

    msg = "Coordinate frame error. Detail: %(detail)s"

    def __init__(self, detail: Any = ""):
        DodoException.__init__(self, detail=detail)


class VarTypeErrorException(DodoException):
    """
    变量类型不符合要求
    """

    def __init__(self, name: str, given_var: Any):
        if not isinstance(given_var, str):
            given_var = str(type(given_var))

        super().__init__(name=name, given_var=given_var)

    msg = "Variable %(name)s type error, Given: %(given_var)s."


class ModuleInterfaceMissingException(DodoException):
    """框架依赖的模块接口未注册（如 ModelManager promote caller）。"""

    msg = (
        "Required module interface is missing. "
        "Module: %(module)s, Interface: %(interface)s. %(detail)s"
    )

    def __init__(self, module: str = "", interface: str = "", detail: Any = ""):
        DodoException.__init__(self, module=module, interface=interface, detail=detail)


class AlgoConformanceException(DodoException):
    """
    算法实现不符合框架契约。

    Learner/Agent 二次开发规范校验，以及算法发烟测试中对契约项的统一断言，均使用本异常。
    发烟测试捕获 AlgoConformanceException 即可判定算法接入未通过规范性检查。
    """

    msg = (
        "Algorithm conformance error. "
        "Component: %(component)s, Contract: %(contract)s, Detail: %(detail)s"
    )

    def __init__(self, component: str = "", contract: str = "", detail: Any = ""):
        DodoException.__init__(self, component=component, contract=contract, detail=detail)


class InterfaceMissingException(AlgoConformanceException):
    """必须实现的接口未实现，子类仍是抽象方法。"""

    msg = (
        "Algorithm conformance error: missing required interface. "
        "Component: %(component)s, Contract: %(contract)s, Detail: %(detail)s"
    )

    def __init__(self, component: str = "", contract: str = "", detail: Any = ""):
        AlgoConformanceException.__init__(
            self, component=component, contract=contract, detail=detail
        )


class ContractViolationException(AlgoConformanceException):
    """接口返回值不满足契约要求（类型/形状/语义）。"""

    msg = (
        "Algorithm conformance error: contract violation. "
        "Component: %(component)s, Contract: %(contract)s, Detail: %(detail)s"
    )

    def __init__(self, component: str = "", contract: str = "", detail: Any = ""):
        AlgoConformanceException.__init__(
            self, component=component, contract=contract, detail=detail
        )


class FunctionalityException(AlgoConformanceException):
    """接口功能不可用，端到端调用失败或行为不符合预期。"""

    msg = (
        "Algorithm conformance error: functionality check failed. "
        "Component: %(component)s, Contract: %(contract)s, Detail: %(detail)s"
    )

    def __init__(self, component: str = "", contract: str = "", detail: Any = ""):
        AlgoConformanceException.__init__(
            self, component=component, contract=contract, detail=detail
        )


class ArtifactsArchiveException(DodoException):
    """任务产物登记、归档或查询不满足归档契约。"""

    msg = "Artifacts archive error. Operation: %(operation)s, Detail: %(detail)s"

    def __init__(self, operation: str = "", detail: Any = ""):
        DodoException.__init__(self, operation=operation, detail=detail)
