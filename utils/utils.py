import operator
import sys
# from argparse import ArgumentParsern
from functools import reduce
from types import SimpleNamespace
from typing import Dict, Optional, Union

import yaml


def get_from_dict(input_dict, map_list):
    return reduce(operator.getitem, map_list, input_dict)


def set_in_dict(input_dict, map_list, value):
    get_from_dict(input_dict, map_list[:-1])[map_list[-1]] = value


class ConfigParser:
    def __init__(self) -> None:
        self._config = None

    @property
    def config(self) -> Dict[str, Dict[str, Union[str, int, float]]]:
        return self._config

    def print_config(self):
        print("\n### CONFIG ###\n")
        print(self._config)
        print("\n#############\n")

    def add_config(self, path: str) -> None:
        with open(path) as f:
            self._config = yaml.safe_load(f)

    def add_sigopt_hyperparams(self, params: Dict[str, Union[str, int, float]]) -> None:
        assert (
            self._config is not None
        ), "Config should be provided before adding SigOpt hyperparameters."
        for k, v in params.items():
            print(k, v)
            keys = k.split(".")
            set_in_dict(self._config, keys, v)


def dict_to_namespace(d):
    # operator.attrgetter('model.depth')(namespace)
    x = SimpleNamespace()
    _ = [
        setattr(x, k, dict_to_namespace(v)) if isinstance(v, dict) else setattr(x, k, v)
        for k, v in d.items()
    ]
    return x
