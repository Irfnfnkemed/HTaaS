from dataclasses import dataclass

"""
This module provides the Counter class, which is used to handle the dynamic changes to the accumulation step size.
"""

@dataclass
class Counter:
    count: int = 0
    accum: int = 1
    
    def step(self):
        self.count += 1
        
    def reset(self):
        self.count = 0
    
    def set_accum(self, accum: int):
        """
        Set the accumulation step to the given value. This operation is only allowed when the counter's count is a multiple of the current accumulation step.

        Args:
            accum: The new accumulation step.
        """
        assert self.need_update
        self.count = self.count // self.accum * accum
        self.accum = accum
        
    @property
    def need_update(self) -> bool:
        """
        Whether the training needs to update parameters now.
        """
        return self.count % self.accum == 0
    
    @property
    def iter_count(self) -> int:
        """
        Calculate the number of completed accumulation steps.
        """
        return self.count // self.accum
    
    