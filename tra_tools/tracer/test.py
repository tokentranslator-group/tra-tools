# $ python3 -m tra_tools.tracer.test
from tra_tools.tracer import Tracer, func
import math
import numpy as np


# the func to test:
def f(x):
    y = math.sin(x)
    y = func(y)
    z = np.convolve(np.ones(3), np.ones(3))
    return y


if __name__ == "__main__":
    print("starting:")
    
    tracer = Tracer(
        show_source=True, 
        show_call=True,
        show_line=True,
        show_return=False,

        # include_modules=["test"],
        # include_funcs=["func"],
        exclude_funcs=["<listcomp>"],
        # exclude_modules=["tracer"],

        show_global_modules=False,

        show_name_module=True,
        show_name_func=True,
    )
    f1 = tracer(f)
    f1(3)
    print("done, see result in:", tracer.filename)
