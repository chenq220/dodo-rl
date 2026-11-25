import torch.nn as nn
from typing import Tuple, Any, List
from prettytable import PrettyTable, HRuleStyle


def print_model_info(LOG, agent_net, algo_type):
    if agent_net is None:
        LOG.info(f"智能体 [{algo_type}] 不使用可训练网络。")
        return

    def _extract_network_details() -> Tuple[List[str], List[List[Any]], int]:
        """
        修复后的网络层级数据提取函数
        """
        headers = ["Layer Name", "Layer Type", "In Dim", "Out Dim", "Weight Shape",
                   "Params"]
        rows = []
        total_params = 0

        for name, module in agent_net.named_modules():
            # 跳过最外层的容器（也就是模型本身）
            if name == "" or len(list(module.children())) > 0:
                continue

            layer_type = module.__class__.__name__
            in_features, out_features, weight_shape, layer_params = "-", "-", "-", 0

            # 1. 全连接层解析
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                layer_params = sum(p.numel() for p in module.parameters())
                weight_shape = str(list(module.weight.shape))

            # 2. 循环层解析
            elif isinstance(module, (nn.GRU, nn.LSTM, nn.GRUCell, nn.LSTMCell)):
                in_features = module.input_size
                out_features = module.hidden_size
                layer_params = sum(p.numel() for p in module.parameters())
                weight_shape = f"H_size: {module.hidden_size}"

            # 3. 其他参数层（如 Conv2d 等）
            else:
                layer_params = sum(p.numel() for p in module.parameters())
                # 如果有权重则显示形状
                if hasattr(module, 'weight') and module.weight is not None:
                    weight_shape = str(list(module.weight.shape))

            total_params += layer_params

            rows.append([name, layer_type, in_features, out_features, weight_shape, layer_params])

        return headers, rows, total_params

    def _print_pretty_table(headers: List[str], rows: List[List[Any]], total_params: int):
        """
        使用全新的 HRuleStyle 规避过时警告，渲染高颜值表格看板
        """
        table = PrettyTable()
        table.field_names = headers

        # 布局对齐
        table.align["Layer Name"] = "l"
        table.align["Layer Type"] = "l"
        table.align["In Dim"] = "c"
        table.align["Out Dim"] = "c"
        table.align["Weight Shape"] = "c"
        table.align["Params"] = "r"

        for row in rows:
            formatted_row = list(row)
            if isinstance(formatted_row[-1], int):
                formatted_row[-1] = f"{formatted_row[-1]:,}"
            table.add_row(formatted_row)

        table.hrules = HRuleStyle.ALL

        LOG.info(f'\n{"═" * 20} Network structure [{algo_type.upper()}] {"═" * 20}'
                 f'\n{table}'
                 f'\nTotal Trainable parameters: {total_params:,}'
                 f'\n{"═" * 72}')

    headers_, rows_, total_params_ = _extract_network_details()
    _print_pretty_table(headers_, rows_, total_params_)
