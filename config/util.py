import ruamel.yaml as yaml
import argparse
import os
import warnings


def load_conf(path: str = None, method: str = None, dataset: str = None):
    """
    加载配置文件函数

    参数：
      - path: 配置文件路径。如果为 None，则使用默认路径构造配置文件路径。
      - method: 使用的方法名称。如果 path 为 None，则必须提供此参数。
      - dataset: 数据集名称。如果 path 为 None，则必须提供此参数。

    返回：
      - conf: 加载的配置文件，并转换为 argparse.Namespace 对象，便于通过属性访问配置参数。
    """
    # 构造默认的配置文件目录，当前工作目录下的 "config" 文件夹
    # dir = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/config'
    dir = os.getcwd() + '/config/'
    # 根据方法和数据集名称构造配置文件路径，例如: config/{method}/{method}_{dataset}.yaml
    path = os.path.join(dir, method, method + '_' + dataset + ".yaml")
    # 忽略 yaml.UnsafeLoaderWarning 警告
    warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

    # 如果 path, method, 或 dataset 均为 None，则抛出异常
    if path is None and method is None:
        raise KeyError("必须提供配置文件路径或者方法名称。")
    if path is None and dataset is None:
        raise KeyError("必须提供配置文件路径或者数据集名称。")

    # 如果 path 为 None，则从项目的上一级目录构造配置文件路径
    if path is None:
        dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
        if method in ["link", "lpa"]:
            path = os.path.join(dir, method, method + ".yaml")
        else:
            path = os.path.join(dir, method, method + '_' + dataset + ".yaml")
        if not os.path.exists(path):
            raise KeyError("The configuration file is not provided.")

    # 读取配置文件内容
    conf = open(path, "r").read()
    # 加载 YAML 文件，解析为 Python 字典
    conf = yaml.load(conf)

    # 导入 nni 模块，用于超参数自动调优
    import nni
    # 如果当前试验不为 STANDALONE，则获取下一组超参数，并覆盖配置文件中对应的值
    if nni.get_trial_id() != "STANDALONE":
        par = nni.get_next_parameter()
        # 遍历配置字典，若子项为字典，则检查是否有需要被更新的键
        for i, dic in conf.items():
            if isinstance(dic, dict):
                for a, b in dic.items():
                    for x, y in par.items():
                        if x == a:
                            conf[i][a] = y
            # 同时检查顶级键是否需要被覆盖
            for x, y in par.items():
                if x == i:
                    conf[i] = y

    # 将字典转换为 argparse.Namespace 对象，方便通过 .key 方式访问配置参数
    conf = argparse.Namespace(**conf)
    return conf


def save_conf(path, conf):
    """
    保存配置文件函数

    参数：
      - path: 保存配置文件的路径。
      - conf: 配置对象（通常为 argparse.Namespace 对象），内部存储了所有配置参数。

    功能：
      - 将配置对象转换为字典，并以 YAML 格式保存到指定路径。
    """
    with open(path, "w", encoding="utf-8") as f:
        # vars(conf) 将 Namespace 对象转换为字典，再通过 yaml.dump 写入文件
        yaml.dump(vars(conf), f)
