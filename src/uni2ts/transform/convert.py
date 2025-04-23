from dataclasses import dataclass
import numpy as np

from ._base import Transformation


@dataclass
class DataTypeConverter(Transformation):
    field: str
    datatype: type = np.float32

    def __call__(self, data_entry):
        arr = data_entry[self.field]
        arr = arr.astype(self.datatype)
        
        data_entry[self.field] = arr

        return data_entry