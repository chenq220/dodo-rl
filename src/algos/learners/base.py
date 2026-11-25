from __future__ import annotations
import copy
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
from abc import ABC, abstractmethod
from dataclasses import fields, is_dataclass
from typing import Dict, Generic, List, Any, Optional, Callable, Tuple, TYPE_CHECKING, Union, TypeVar

from prettytable import PrettyTable, HRuleStyle
from torch.nn import Module

from common.exception import KeyErrorException, ValueErrorException
from scheduler.artifact import ArtifactCallerSupport, ArtifactType, ensure_output_parent
from common.logger import init_logger
from config import AgentOutputType, LearnerConfig
from envs.base import BaseEnvironment
from algos.agents import Agent
from sample.sample_transfer import EpisodeTransition
from sample.model import EpisodeBatch
from utils.print_model import print_model_info

from common.types import ModelName
from .batch_schema import DEFAULT_EPISODE_SCHEMA

if TYPE_CHECKING:
    from common.types import nn, FilePath, NetWeight, ModelSource

LOG = init_logger('learner')
FieldsT = TypeVar("FieldsT")


def algo_config_to_updates(algo_config: Any) -> Dict[str, Any]:
    """把 algo_config 转为 dynamic_modify_config 的 dict 入参（调用方辅助，非 Learner 接口）。"""
    if isinstance(algo_config, dict):
        return dict(algo_config)
    if algo_config is None:
        return {}
    if hasattr(algo_config, "to_dict") and callable(algo_config.to_dict):
        return algo_config.to_dict()
    if is_dataclass(algo_config):
        return {field.name: getattr(algo_config, field.name) for field in fields(algo_config)}
    raise TypeError(
        f"algo_config_to_updates expects dict or algo_config dataclass, "
        f"given {type(algo_config).__name__}."
    )


class Learner(ArtifactCallerSupport, ABC, Generic[FieldsT]):
    """
    学习器基类，负责算法网络构建、参数更新、模型保存和策略网络导出。

    Learner 面向算法包封装：可以内部持有 mixer、critic、target network 等多个网络，
    但必须通过 agent_net 暴露一个可被业务 Agent 克隆的策略网络。
    """

    supported_action_spaces = ("discrete",)
    batch_schema = DEFAULT_EPISODE_SCHEMA

    def __init__(
            self,
            algo_type: str,
            config: LearnerConfig,
            agent: 'Agent',
            env: BaseEnvironment,
            device: Optional[Union[torch.device, str]],
            net_source: ModelSource,
            model_file_path: Dict[str, FilePath] = None,
            caller: Callable[[bool], Dict[str, NetWeight]] = None,
            uploader: Callable[[Dict[str, 'nn.Module']], None] = None,
            **kwargs
    ):
        """
        :param algo_type: 算法类型标识（例如 "qmix", "mppo"）。
        :param config: Learner算法配置。
        :param agent: 智能体实例
        :param env: 环境实例
        :param device: 计算设备
        :param net_source: 初始化权重来源：caller ：从外部进程的函数中获取网络权重，file：从文件加载网络权重，blanco：加载空网络权重
        :param model_file_path: 初始化模型file源指定模型文件路径
        :param caller: 初始化模型获取网络权重的外部函数
        :param uploader: 将模型上传到外部程序
        :param kwargs: 算法拓展参数
        """
        LOG.info(f'Learner name is [{config.name}]')
        LOG.info(f'Learner algo is [{algo_type}]')
        LOG.info(f'Learner device is [{device}]')
        self.name = config.name
        # 产物 register/put caller；由 Role 接线，未绑定时跳过。
        self._init_artifact_callers()
        # 算法类型标识，对应注册表中的 learner 类型。
        self.algo_type = algo_type
        # Learner算法配置，包含算法网络、优化器和运行期维度参数。
        self.config = config
        # 当前训练使用的业务 Agent，Learner 可读取其动作表、输入维度和策略网络结构。
        self.agent = agent
        from envs.action.spaces import require_action_space_support
        if getattr(agent, "action_space", None) is not None:
            require_action_space_support(self, agent.action_space, label=f"Learner {algo_type} ({self.name})")
        assert self.agent.config.algo_config.use_net, "None network agent can not be learner agent."
        # 配对 Agent 的输出语义等字段由 Learner 算法配置回填，入口无需手动对齐。
        self._sync_paired_agent_config(config, agent)
        # torch 计算设备，所有学习器网络应迁移到该设备。
        self.device = device if isinstance(device, str) else torch.device(device)
        # 环境实例，提供状态/观测/动作维度以及必要的业务接口。
        self.env = env
        # 可选模型上传回调，供分布式或外部模型池集成。
        self.uploader = uploader
        # 学习器内部所有需要 train/eval 切换的网络列表，由具体算法构造。
        self.net_list: tuple[Module, ...] = self._setup_network(net_source, model_file_path, caller, config, **kwargs)
        # 设置模式
        for net in self.net_list:
            net.train()
        # Learner 自持有训练状态快照；外部任务控制只下达“保存/恢复”指令，不关心算法内部状态结构。
        # 快照内容完全由子类通过 snapshot_state/restore_state 决定，基类只负责调度与深拷贝隔离。
        self._model_snapshots: Dict[str, Any] = {}
        # Learner 本地代次；TaskManager 重启任务后会在学习令牌中下发更大的 generation。
        self.generation = 0
        self.clone_models_snapshot()
        self._print_auxiliary_network_info()
        self._print_algo_config(config)

    def register_stage_intents(self) -> None:
        self.register_artifact_intent(ArtifactType.MODEL)

    @staticmethod
    def _print_algo_config(config: LearnerConfig) -> None:
        """使用PrettyTable打印入口最终组织完成的具体算法超参。"""
        algo_config = config.algo_config
        assert algo_config is not None, f"LearnerConfig[{config.name}].algo_config must not be None."
        assert is_dataclass(algo_config), (
            f"LearnerConfig[{config.name}].algo_config must be a dataclass, "
            f"given={type(algo_config).__name__}"
        )

        table = PrettyTable()
        table.field_names = ["Parameter", "Value", "Type"]
        table.align["Parameter"] = "l"
        table.align["Value"] = "l"
        table.align["Type"] = "l"
        table.hrules = HRuleStyle.ALL
        for config_field in fields(algo_config):
            value = getattr(algo_config, config_field.name)
            table.add_row([
                config_field.name,
                str(value),
                type(value).__name__,
            ])

        LOG.info(
            f'\n{"═" * 20} Learner algorithm config '
            f'[{config.name}:{config.algo_type}] {"═" * 20}\n{table}'
        )

    def _print_auxiliary_network_info(self) -> None:
        """打印 Learner 内除策略 agent_net 之外的 mixer、critic 等辅助网络结构。"""
        agent_net = self.agent_net
        printed_model_ids = {id(agent_net)}
        for model_name, model in self.models.items():
            if model_name == "agent_net" or model is None or id(model) in printed_model_ids:
                continue
            if not isinstance(model, Module):
                LOG.warning(f"Skip non-module learner model info: learner={self.name}, model_name={model_name}")
                continue
            print_model_info(LOG, model, f"{self.algo_type}:{model_name}")
            printed_model_ids.add(id(model))

    @staticmethod
    def _sync_paired_agent_config(learner_config: LearnerConfig, agent: Agent) -> None:
        """
        用 Learner 算法配置回填配对 Agent 的运行语义，避免入口重复写桥接字段。

        当前只同步 agent_output_type(q / pi_logits)：由 LearnerConfig.algo_config 决定，
        AgentConfig 侧保留通用默认值即可，Learner 初始化拿到 agent 后自动对齐。
        """
        algo_config = learner_config.algo_config
        if algo_config is None or not hasattr(algo_config, "agent_output_type"):
            return
        required = AgentOutputType(algo_config.agent_output_type).value
        fixed = getattr(agent, "fixed_output_type", None)
        if fixed is not None and required != fixed:
            raise ValueError(
                f"Learner {learner_config.algo_type} requires {required!r}, "
                f"but Agent {agent.agent_type} has fixed output {fixed!r}"
            )
        if agent.config.agent_output_type != required:
            LOG.debug(
                f"Sync paired agent_output_type: {agent.config.agent_output_type} -> {required} "
                f"(learner={learner_config.name}, agent={agent.name})"
            )
            agent.config.agent_output_type = required
        # 执行期 MAC 在 Agent 构造时已建好；仅改 config 不会更新 mac.forward 的分支，需同步 mac。
        mac = getattr(agent, "mac", None)
        if mac is not None and getattr(mac, "agent_output_type", None) != required:
            mac.agent_output_type = required
            mac_args = getattr(mac, "args", None)
            if mac_args is not None and hasattr(mac_args, "agent_output_type"):
                mac_args.agent_output_type = required

    def _setup_network(self, agent_net_source, model_file_path, caller, config, **kwargs) -> tuple[Module, ...]:
        """通用的网络构建与权重恢复控制流。"""
        # 从外部程序获取网络权重
        if agent_net_source == 'caller':
            assert caller, "agent_net_source='caller' requires caller."
            LOG.info(f"Setup network caller is: {caller}")
            caller_checkpoint_dict = caller(agent_only=False)
            return self.build_network_from_checkpoint(
                checkpoint=caller_checkpoint_dict,
                learner_config=config,
                **kwargs
            )
        # 从文件获取网络权重
        elif agent_net_source == 'file':
            assert model_file_path, 'agent_net_source is file, needs model_file_path'
            LOG.info(f"Setup network model file: {model_file_path}")
            file_checkpoint_dict = {}
            for model_name, file_path in model_file_path.items():
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Model file not found at: {file_path}")
                file_checkpoint = torch.load(file_path, map_location=self.device)
                file_checkpoint_dict[model_name] = file_checkpoint
            return self.build_network_from_checkpoint(
                checkpoint=file_checkpoint_dict,
                learner_config=config,
                **kwargs
            )
        # 创建空网络
        elif agent_net_source == 'blanco':
            LOG.info("Build a blanco network.")
            return self.build_empty_network(
                learner_config=config,
                **kwargs
            )

        else:
            raise ValueError(f"Unsupported network source: {agent_net_source}")

    def upload_models(self) -> Any:
        """ 将模型上传给其他程序 """
        if self.uploader is None:
            raise RuntimeError("No uploader specified.")
        return self.uploader(self.models)

    def clone_models_snapshot(self, snapshot_name: str = "default") -> None:
        """
        保存一份当前训练状态快照。

        调用子类 snapshot_state() 收集状态、深拷贝隔离、按名字存盘、写日志。
        子类无需关心何时调用本方法（任务重启、阶段结束时会自动触发），只需实现 snapshot_state。
        """
        state = self.snapshot_state()
        # 基类统一深拷贝，保证存盘与活模型互不引用，重复恢复也不会污染历史快照。
        self._model_snapshots[snapshot_name] = copy.deepcopy(state)
        LOG.info(f"Learner snapshot saved: learner={self.name}, snapshot={snapshot_name}")

    def restore_models_snapshot(self, snapshot_name: str = "default") -> None:
        """
        从内部快照恢复训练状态。

        取出存盘、深拷贝一份传给子类 restore_state、写日志。
        若快照不存在直接抛错，避免任务重启时静默沿用错误模型。
        """
        if snapshot_name not in self._model_snapshots:
            raise KeyErrorException(
                key=snapshot_name,
                container=f"Learner[{self.name}]._model_snapshots",
            )
        # 传给子类的是存盘的拷贝，子类可以放心就地修改而不影响历史快照。
        self.restore_state(copy.deepcopy(self._model_snapshots[snapshot_name]))
        LOG.info(f"Learner snapshot restored: learner={self.name}, snapshot={snapshot_name}")

    def restore_models_snapshot_for_generation(self, generation: int, snapshot_name: str = "default") -> None:
        """
        任务重启时由 Role 调用：立即恢复快照并写入本地 generation。

        采样侧 Agent 必须在下一局开始前拿到恢复后的策略网络，不能等到下一次 update 时再恢复。
        这是恢复主路径；learn_update 内部的 _sync_generation 只是兜底。
        """
        self.restore_models_snapshot(snapshot_name=snapshot_name)
        self.generation = int(generation)
        LOG.warning(
            f"Learner generation restored by role: learner={self.name}, generation={self.generation}"
        )

    def reset_generation(self) -> None:
        """阶段切换时把 Learner 本地代次重置为 0，下一阶段使用新的 TaskManager 代次序列。"""
        self.generation = 0
        LOG.info(f"Learner generation reset: learner={self.name}")

    def learn_update(self, batch_data: List[EpisodeTransition], generation: int = 0) -> Dict[str, Any]:
        """
        框架内部学习入口。

        RolloutLearnWorker 调用该接口并透传 TaskManager 令牌代次。二次开发者只需要实现
        update(batch_data)，不需要理解代次和快照恢复协议。
        """
        if generation > self.generation:
            self.restore_models_snapshot()
            self.generation = generation
            LOG.info(
                f"Learner generation advanced and snapshot restored (fallback): "
                f"learner={self.name}, generation={self.generation}"
            )
        if self.batch_schema is not None and batch_data:
            transition_type = type(batch_data[0])
            if any(type(episode) is not transition_type for episode in batch_data):
                raise TypeError(f"Learner {self.name} received incompatible trajectory types")
            context = SimpleNamespace(agent=self.agent, device=torch.device(self.device))
            batch_data = transition_type._assemble_batch(
                batch_data, learner_name=self.name, schema=self.batch_schema, context=context,
            )
        return self.update(batch_data)

    @property
    @abstractmethod
    def models(self) -> Dict[ModelName, 'nn.Module']:
        raise NotImplementedError('子类必须实现 models 属性')

    @abstractmethod
    def build_empty_network(self, learner_config, **kwargs) -> Tuple['nn.Module', ...]:
        """
        使用空网络初始化或重新初始化网络
        :param learner_config: 算法特有网络配置
        :param kwargs: 其他参数输入，子类初始化类时指定
        :return: None
        """
        raise NotImplementedError('子类必须实现 build_empty_network 方法')

    @abstractmethod
    def build_network_from_checkpoint(self, checkpoint: Dict[str, NetWeight],
                                      learner_config, **kwargs) -> Tuple['nn.Module', ...]:
        """
        加载网络权重文件初始化或重新初始化网络
        :param checkpoint: 算法所需各网络权重值
        :param learner_config: 检查checkpoint的算法特有网络是否满足维度要求
        :param kwargs: 其他参数输入，子类初始化类时指定
        :return: None
        """
        raise NotImplementedError('子类必须实现 build_empty_network 方法')

    @abstractmethod
    def _offline_net_weights(self) -> Dict[str, Dict]:
        """ 获取其他非agent网络权重，如果没有返回空字典 """
        raise NotImplementedError('子类必须实现 _offline_net_weights 方法')

    @abstractmethod
    def update(self, batch_data: EpisodeBatch[FieldsT]) -> Dict[str, Any]:
        """
        算法更新网路权重
        :param batch_data: 根据当前 Learner 的 batch_schema 组装的训练数据。
        :return: 包含此次更新关键指标的字典（如 loss 值），便于写入日志。
        """
        raise NotImplementedError('子类必须实现 update 方法')

    @abstractmethod
    def snapshot_state(self) -> Any:
        """
        子类实现：返回当前需要保存的全部训练状态。

        返回值会被基类整体深拷贝后存盘，因此子类可以直接返回对活模型 state_dict、计数器、
        归一化器等的引用，无需自己再做深拷贝。快照必须覆盖所有会影响下一次 update 结果的状态，
        否则任务重启后曲线会分叉：
        - 全部网络权重（含 target / old / mixer 等辅助网络）的 state_dict；
        - 全部优化器的 state_dict（动量、学习率调度状态）；
        - 训练计数器、动作归一化器、回报归一化器等非网络训练状态。

        建议返回 dict，key 用算法自有命名空间前缀以防冲突，例如 {"epymarl_training_steps": ...}。
        """
        raise NotImplementedError("子类必须实现 snapshot_state 方法")

    @abstractmethod
    def restore_state(self, state: Any) -> None:
        """
        子类实现：从 snapshot_state 返回值恢复训练状态。

        传入的 state 是存盘的深拷贝，子类可以就地读取/修改。恢复动作应与 snapshot_state 一一对应：
        load_state_dict 回全部网络与优化器、写回全部计数器与归一化器。
        """
        raise NotImplementedError("子类必须实现 restore_state 方法")

    @property
    @abstractmethod
    def agent_net(self) -> 'nn.Module':
        """
        agent策略网络，供外部使用
        :return: agent策略网络模型
        """
        raise NotImplementedError('子类必须实现 agent_net 属性')

    def save_models(
            self,
            save_model_path: FilePath,
            save_model_name: Dict[ModelName, str],
    ) -> List[FilePath]:
        """
        将 models 中的网络权重保存为文件。

        调用方（TrainTaskManager）负责生成文件名；本方法只按 save_model_name 映射落盘 state_dict。
        子类仅在需要额外保存 models 未列出的网络、或采用非 state_dict 格式时覆盖本方法。
        """
        saved_files: List[FilePath] = []
        for model_name, model in self.models.items():
            if model is None:
                raise ValueErrorException(
                    name=f"Learner[{self.name}].models['{model_name}']",
                    expected="nn.Module",
                    given=None,
                )
            if model_name not in save_model_name:
                raise KeyErrorException(
                    key=model_name,
                    container=f"save_model_name (Learner[{self.name}])",
                )
            file_path = os.path.join(save_model_path, save_model_name[model_name])
            ensure_output_parent(file_path)
            torch.save(model.state_dict(), file_path)
            self.put_artifact_file(ArtifactType.MODEL, file_path)
            saved_files.append(file_path)
        LOG.info(f"Learner models saved: learner={self.name}, files={saved_files}")
        return saved_files

    def dynamic_modify_config(self, updates: Optional[Dict[str, Any]]) -> None:
        """
        运行期动态切换算法超参（默认实现：增量替换）。

        只接受 dict：键为 algo_config 字段名，仅覆盖传入字段，其余保持不变。
        子类覆盖时应先 ``super().dynamic_modify_config(updates)``，再同步框架内部状态。

        :param updates: 待更新的超参增量；None 或空 dict 时不修改。
        """
        if not updates:
            return
        if not isinstance(updates, dict):
            raise TypeError(
                f"dynamic_modify_config expects dict updates, given {type(updates).__name__}."
            )
        target = self.config.algo_config
        assert target is not None, (
            f"Learner[{self.name}].config.algo_config is None; cannot apply dynamic config."
        )
        for key, value in updates.items():
            if not hasattr(target, key):
                raise AttributeError(
                    f"{type(target).__name__} has no algo config field '{key}'."
                )
            setattr(target, key, value)
