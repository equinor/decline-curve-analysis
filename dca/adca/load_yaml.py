"""
Custom .yaml loading that removes the 1024 character limit on simple keys.
"""

import yaml
from yaml.composer import Composer
from yaml.constructor import SafeConstructor
from yaml.parser import Parser
from yaml.reader import Reader
from yaml.resolver import Resolver
from yaml.scanner import Scanner


class CustomScanner(Scanner):
    """Scanner with 1024-char limit on simple keys removed."""

    def stale_possible_simple_keys(self):
        # https://github.com/yaml/pyyaml/blob/69c141adcf805c5ebdc9ba519927642ee5c7f639/lib/yaml/scanner.py#L279
        return None


class CustomSafeLoader(
    Reader, CustomScanner, Parser, Composer, SafeConstructor, Resolver
):
    """SafeLoader which inherits from CustomScanner instead of Scanner."""

    def __init__(self, stream):
        Reader.__init__(self, stream)
        CustomScanner.__init__(self)
        Parser.__init__(self)
        Composer.__init__(self)
        SafeConstructor.__init__(self)
        Resolver.__init__(self)


def yaml_safe_load(stream):
    """Load a .yaml file using `yaml.load()`."""

    return yaml.load(stream, CustomSafeLoader)
