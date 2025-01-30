import sys
import importlib

class intra_transformers:
    def __init__(self, target_module_name):
        self.target_module_name = target_module_name
        self.target_module = None

    def __getattr__(self, name):
        print(name)
        if self.target_module is None:
            self.target_module = importlib.import_module(self.target_module_name)

        if name == 'Trainer':
            from .job import Trainer
            return Trainer
        try:
            return getattr(self.target_module, name)
        except AttributeError:
            try:
                submodule_name = f"{self.target_module_name}.{name}"
                submodule = importlib.import_module(submodule_name)
                return submodule
            except ImportError:
                raise AttributeError(f"module '{self.target_module_name}' has no attribute '{name}'")

sys.modules['intra_transformers'] = intra_transformers('transformers')
