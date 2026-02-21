# Interface class to outline methods needed to be implemented for incremental context update support.
# Based on the IncTNP codebase
from abc import ABC, abstractmethod
import torch
import torch.distributions as td

# General inc updates
class IncUpdateEff(ABC):
    @abstractmethod
    def init_inc_structs(self):
        raise NotImplementedError
    
    @abstractmethod
    def update_ctx(self, xc: torch.Tensor, yc: torch.Tensor):
        raise NotImplementedError
    
    # @abstractmethod
    # def repeat_ctx(self, repeat_times: int, persist_small: bool=False):
    #     raise NotImplementedError

    @abstractmethod
    def query(self, xt: torch.Tensor) -> td.Normal:
        raise NotImplementedError