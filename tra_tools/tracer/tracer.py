import os
import sys
from sys import settrace
import inspect
import math
import re


class Tracer():
    '''To trace some obj.call internally i.e. with use of import system.
    Usage::

    >>> from tra_tools.tracer import Tracer

    >>> tracer = Tracer(
    >>>    show_source=True,

    >>>    # events to show:
    >>>    show_call=True,
    >>>    show_line=True,
    >>>    show_return=False,

    >>>    # inclusion/exclusion manually:
    >>>    # include_modules=["test"],
    >>>    # include_funcs=["func"],
    >>>    exclude_funcs=["<listcomp>"],
    >>>    # exclude_modules=["tracer"],

    >>>    show_global_modules=False,  # from sys.path

    >>>    show_name_module=True,  # show module name
    >>>    show_name_func=True,   # show func name

    >>>    separator="    ->   "  # to separate module and func names
    >>> )
    >>> test = tracer(test_opt)

    >>> # run with same args as test_opt:
    >>> test(steps=1, t1=0.3, lr=0.1, deg=3)
    '''

    def __init__(self,
                 include_funcs=[],
                 exclude_funcs=[],
                 include_modules=[],
                 exclude_modules=[],
                 
                 show_source=False, show_call=False, show_return=True,
                 show_line=False,
                 show_global_modules=False,
                 show_name_func=True,
                 show_name_module=True,
                 separator="->",
                 to_file=True, filename="/tmp/tra_tools.tracer.log.txt"):
        self.clear_context()

        self.include_funcs = include_funcs
        self.include_modules = include_modules
        self.exclude_modules = exclude_modules
        self.exclude_funcs = exclude_funcs

        self.file = None
        self.to_file = to_file
        self.filename = filename

        self.show_source = show_source
        self.show_call = show_call
        self.show_return = show_return
        self.show_line = show_line
        self.show_global_modules = show_global_modules

        self.show_name_func = show_name_func
        self.show_name_module = show_name_module

        self.separator = separator
        # self.lcode = lcode
        # self.lnumber = lnumber

    def _cond_include(self, module_name, func_name):

        '''case user would specify some names explicitly'''

        if (module_name in self.exclude_modules
            or func_name in self.exclude_funcs):
            return False
        elif ((len(self.include_modules) > 0
               and module_name not in self.include_modules) or
              (len(self.include_funcs) > 0
               and func_name not in self.include_funcs)):
            return False
        else:
            return True

    def _cond_mod_glob(self, co_filename):
        
        return (
            "site-packages" in co_filename
            or "python" in co_filename
            or "internals" in co_filename
            or "<frozen" in co_filename
            or "<string>" in co_filename)

    def show(self, out):
        if not self.to_file:
            print(out, end="\n")
        else:
            self.file.write(out+"\n")
                
    def append(self, f_code, lineno, out=""):

        '''Add an info about the f_code to the out,
        initiated with `out` and found on the `lineno`.'''

        co_filename = f_code.co_filename

        func_name, module_name = self._get_names(f_code)
    
        if self._cond_include(module_name, func_name):
            # check if the `f_code` location is in 
            # local or global (sys.path) module
            # and if we should add its info:
            if (not self._cond_mod_glob(co_filename)
                or self.show_global_modules):
                indent = 1*self.indent * " " 
                out += self.append_name(f_code, out=indent)
                if self.show_source:
                    out += self.append_source(f_code, lineno, out="\n"+indent)
                self.show(out)
                # print(out, end="\n")
        return out

    def append_source(self, f_code, lineno, out=""):
        '''Find and add source to `out`'''
        try:
            lcode, lnumber = inspect.getsourcelines(f_code)
            out += re.sub("\s{2,}", "", lcode[lineno-lnumber])
        except:
            pass
            # print("ERROR inspect.getsourcelines")
        return out

    def append_name(self, f_code, out=""):
        
        '''Add the `f_code` module name and funct name to the output'''

        func_name, module_name = self._get_names(f_code)

        if self.show_name_func and self.show_name_module:
            out += "|" + module_name + self.separator + func_name
        elif self.show_name_func:
            out += func_name
        elif self.show_name_module:
            out += "|" + module_name 
        '''
        if ("site-packages" in co_filename
            or "python" in co_filename):
            out += "global module:: " + module_name + ": " + func_name 
        else:
            out += "local module: " + module_name + ": " + func_name
        '''
        return out
        # print("inspect module_name")
        # print(inspect.getmodule(frame.f_code))

    def _get_names(self, f_code):
        func_name = f_code.co_name
        co_filename = f_code.co_filename
        co_basename = os.path.basename(co_filename)
        module_name, _ = os.path.splitext(co_basename)
        return (func_name, module_name)

    def clear_context(self):
        self.frames = []
        self.lines = []
        self.returns = []
        self.indent = 0

    def __enter__(self, *args, **kwargs):
        self.clear_context()
        if self.to_file:
            self.file = open(self.filename, "w")
        return self

    def __exit__(self, *args):
        if self.file is not None:
            self.file.close()

    def __call__(self, f):

        '''Wrapper for tracing a given function `f`. '''

        def f1(*args, **kwargs):
            
            with self:
                # set up handler:
                settrace(self.trace_handler)
                try:
                    '''
                    l = {}
                    d = {"f": f}
                    d.update(locals())
                    exec('f(*args, **kwargs)', d, l)
                    res = l
                    '''
                    res = f(*args, **kwargs)
                finally:
                    # release handler:
                    settrace(None)
                return res
        return f1

    def trace_handler(self, frame, event, arg):

        '''What to do with each frame/line.
        Returning itself is a requirement of `settrace`
        to go in subframes recursively.'''

        lineno = frame.f_lineno
        
        if event == "call":
            self.indent += 1
            if self.show_call:
                self.append(frame.f_code, lineno, out="")
                # if not self.show_return:
                #     print(out, end="\n")

        # self.frames.append(frame)
        if event == "return":
            self.indent -= 1
                    
            # self.returns.append(frame)
            if self.show_return:
                self.append(frame.f_code, lineno, out="")
                
        elif(event == "line"):
            if self.show_line:
                self.append(frame.f_code, lineno, out="")
            # self.lines.append(frame)

        # here is a trick:
        return self.trace_handler


# ================ TESTS ===============:
# FOR some functions to test:
def sin(x):
    r = math.sin(x)
    return r


def square(x):
    return x*x
 

def func(x):
    y = 0
    for i in range(3):
        y += x+2
    tmp = sum([i for i in range(3)])
    z = square(y)+sin(tmp)
    square(z)
    return x + z
# END FOR


def run():
    tracer = Tracer()
    test1 = tracer(func)
    test1(3)
    # import pdb; pdb.set_trace()


if __name__ == "__main__":
    run()
