"""两个 SDK 的动作各自实现，保持当前字段及序列化协议一致。"""
from dodorl_sdk import Action as RLAction
from dodo_sim import Action as SimAction

if __name__ == "__main__":
    action = RLAction("command", {"value": 1}, -2, 0)
    print(SimAction.from_dict(action.to_dict()))
