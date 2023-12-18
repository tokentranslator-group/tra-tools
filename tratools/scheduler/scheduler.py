import numpy as np
from functools import reduce

import sys
try:
    import pyschedule as sch
except:
    
    print("the pyschedule lib requires the python3.6")
    print("You python version is:")
    print(sys.version)
    sch = None

import matplotlib.pyplot as plt


class PySchedulerNotSupportedException(Exception):
    def __init__(self, *args, **kwargs):
        # self.exception = exception
        msg = (
            "\n\nThe `pyschedule` lib requires the `python3.6`"
            + "\nYour python version is:"
            + "\n" + sys.version)
        Exception.__init__(self, msg, **kwargs)


class CpuScheduler():
    def __init__(self, init_entry,
                 target_tasks_params_names=["length", "delay_cost"],
                 dbg=False):
        self.dbg = dbg
        self.ientry = init_entry
        self.columns = self.ientry.attrs_names
        self.columns_targets = self.ientry.attrs_targets
        self.target_tasks_params_names = target_tasks_params_names

    def __call__(self, pack, cpu_count=8):
        self.cpu_count = cpu_count
        with self:
            if self.dbg:
                print("================ running: scheduler.load:")
            df = self.load(pack)
            
            if self.dbg:
                print(df)
                print("\n================ running: scheduler.group_by_named")
            self.group_by_named(df)

            if self.dbg:
                print("\n================ running: scheduler.heuristics(df):")
            self.mk_heuristics(df)

            if self.dbg:
                print("\n================ running: scheduler.mk_tasks(df):")
            self.mk_tasks(df)

            if self.dbg:
                print("\n================ running: scheduler.mk_problem(df):")
            self.mk_problem()

            if self.dbg:
                print("\n================ running: scheduler.solve():")
            self.solve()

            if self.dbg:
                self.plot()

    def __enter__(self):
        '''Clear the `self` state'''

        if hasattr(self, 'gmap_attrs_t_n_st'):
            del self.gmap_attrs_t_n_st
        if hasattr(self, 'heuristics'):
            del self.heuristics
        if hasattr(self, 'tasks'):
            del self.tasks
        if hasattr(self, 'problem'):
            del self.problem
        if hasattr(self, 'tasks_obj'):
            del self.tasks_obj

        return self
            
    def __exit__(self, *args):
        pass

    def load(self, pack):
        '''.
        '''
        df = self.ientry.to_pandas(pack)
        df = df.astype(dict(zip(
            self.columns_targets,
            [np.float]*len(self.columns_targets))))
        # df = df.astype({"dtime": np.float, "goal": np.float})
        
        return df

    def group_by_named(self, df):
        named_attrs = self.columns
        # named_attrs = columns[:-4]
        
        gmap_attrs_t_n_st = dict(zip(
            self.columns_targets,
            list(map(lambda x: {}, self.columns_targets))))
        # as ex: target_attrs = {"dtime":{}, "goal":{}}

        target_attrs_names = self.columns_targets

        for nattr_name in named_attrs:
            
            # factorize by each `nattr` ignoring others `nattrs`:
            nattr_classes = (df.loc[:, [nattr_name]+target_attrs_names]
                             .groupby(nattr_name))

            # firstly apply sum as agregation in the `grouping by`
            # all targets columns independently:
            nattr_classes = nattr_classes.sum()
            
            for tattr_name in target_attrs_names: 

                # then project each target_attribute of factorized table:
                # (i.e for each value of `nattr_name` (i.e. `nattr_value`)
                # for each `tattr_name` find the `tattr`s summed value):
                # (ex: for these like ("nattr_value", "sum of tattr_value"))
                table_nval_stval = dict(zip(list(nattr_classes.index.array),
                                            list(nattr_classes[tattr_name])))

                # and map resulting table from tattr to nattr
                gmap_attrs_t_n_st[tattr_name][nattr_name] = table_nval_stval
                
        if self.dbg:
            print("\ngmap_attrs_t_n_st:")
            print(gmap_attrs_t_n_st)

        self.gmap_attrs_t_n_st = gmap_attrs_t_n_st

    def mk_heuristics(self, df):
        '''
        For estimating each row in the table
        according to the attrs values found in `self.group_by_named`:

        - ``gmap_attrs_t_n_st`` -- map from the tattrs to
        aggregated with use of groupby the sum of nattrs  
        '''
        assert hasattr(self, 'gmap_attrs_t_n_st')
        gmap_attrs_t_n_st = self.gmap_attrs_t_n_st

        named_attrs = self.columns
        target_attrs_names = self.columns_targets

        # estimate the param value for each row in the table
        # which have been built from named attrs only
        # i.e. map each of the `nattr` in a row to its `tattr`
        # aggregated sums given by the param
        # and sum the result for all `nattr`s
        # i.e. sum the `param` value through all values
        # of the named attrs for each row:
        def mapper(row, param="dtime"):
            # row.index here means column name (i.e. nattr):
            # (since axis='columns' used in apply)
            row_converted = list(zip(list(row.index), list(row)))
            row1 = list(map(
                # columnNameVal is name/value
                # for each attribute (nattr) in the row: 
                lambda cNmVal: gmap_attrs_t_n_st[param][cNmVal[0]][cNmVal[1]],
                row_converted))
            return sum(row1)
        # ex: row(cressie, cubic, even) ->
        #    dtime(cressie)+dtime(cubic)+dtime(even)
        # ex: row(cressie, cubic, even) -> goal(cressie)+goal(cubic)+goal(even)

        heuristics = {}
        # for heuristics:
        for tattr_name in target_attrs_names:
            # apply only to named:
            param_expend = df.loc[:, named_attrs].apply(
                lambda x: mapper(x, param=tattr_name),
                axis='columns')
            if self.dbg:
                print("\n%s_expend" % tattr_name)
                print(param_expend)
            param_expend = param_expend.to_numpy()

            # convert to int:
            param_expend1 = (param_expend/param_expend.sum()*100).astype(int)
            if tattr_name == "goal":
                # invert: than less the goal is that better will the result be
                # and hence the dalay_cost must be bigger.
                param_expend1 = param_expend1.max()-param_expend1
        
            if self.dbg:
                print("\ntattr_name:", tattr_name)
                print("param_expend:")
                print(param_expend1)

            heuristics[tattr_name] = param_expend1
        self.heuristics = heuristics
        if self.dbg:
            print("heuristics:")
            print(self.heuristics)

    def mk_tasks(self, df):
        ''' for creating the tasks list'''

        assert hasattr(self, 'heuristics')
        heuristics = self.heuristics

        tasks_names = list(df.index.array)

        tasks_heuristics = list(zip(
            *[(map(int, list(heuristics[tname]))) for tname in heuristics]))

        tasks = dict(zip(
            tasks_names,
            map(lambda x: dict(zip(["length", "delay_cost"], x)),
                tasks_heuristics)))
        
        self.tasks = tasks
        if self.dbg:
            print("tasks:")
            print(tasks)
            
        self.tasks_heuristics = tasks_heuristics
        if self.dbg:
            print("tasks_heuristics:")
            print(tasks_heuristics)
        
    def mk_problem(self):

        ''' for solving the schedule problem'''

        assert hasattr(self, 'tasks')
        tasks = self.tasks
        
        cpu_count = self.cpu_count

        time_length = reduce(
            lambda acc, task_name: acc+tasks[task_name]["length"],
            tasks, 0)
        
        horizon = int(time_length/(cpu_count))
        if self.dbg:
            print("horizon:", horizon)
    
        if sch is not None:
            S = sch.Scenario('selector', horizon=horizon)
        else:
            raise(PySchedulerNotSupportedException())

        tasks_obj = dict([
            (name,
             S.Tasks(
                 str(name)
                 + (("_len_"+str(tasks[name]["length"]))
                    if "length" in tasks[name] else "")
                 + (("_delay_"+str(tasks[name]["delay_cost"]))
                    if "delay_cost" in tasks[name] else "")
                 + "_", **tasks[name]))
            for name in tasks])
        if self.dbg:
            print("tasks_obj:")
            print(tasks_obj)

        self.problem = S
        self.tasks_obj = tasks_obj

    def solve(self):
        assert hasattr(self, 'problem')
        assert hasattr(self, 'tasks_obj')
        
        S = self.problem
        tasks_obj = self.tasks_obj
        cpu_count = self.cpu_count

        # resources:
        R = S.Resources("cpu", num=cpu_count)
        if self.dbg:
            print("R:", R)

        if sch is None:
            raise(PySchedulerNotSupportedException())

        for name in tasks_obj:
            tasks_obj[name] += sch.alt(R)

        sch.solvers.mip.solve(S, msg=1)
        self.S = S

        self.solution = S.solution()
        if self.dbg:
            print("solution:")
            for s in self.solution:
                print(s)
    
    def sort(self, df):
        pass

    def plot(self):
        assert hasattr(self, "tasks")
        tasks = self.tasks

        assert hasattr(self, "S")
        S = self.S
        self.print_solution()
                 
        # [Task(i, duration) for i in range(len(data))]
        # Resorces(num=cpu.count)
        # S.solve()
        if sch is None:
            raise(PySchedulerNotSupportedException())
        sch.plotters.matplotlib.plot(S, img_filename="tmp_selector_schedule.png")
        plt.show()

    def print_solution(self, *args, **kwargs):
        if hasattr(self, "tasks"):
            print("\nScheduler tasks:")

            tasks = self.tasks
            for name in tasks:
                print(
                    "name:", name,
                    "  | length: " if "length" in tasks[name] else "",
                    tasks[name]["length"] if "length" in tasks[name] else "",

                    " | delay_cost: " if "delay_cost" in tasks[name] else "",
                    tasks[name]["delay_cost"] if "delay_cost" in tasks[name] else "")

        if hasattr(self, "solution"):
            print("\nScheduler solution:")
            for cpu_schedule in self.solution:
                print(cpu_schedule)


def test():
    from tratools.zipper import Entry

    scheduler = CpuScheduler(
        init_entry=Entry(
            attrs_names=["A", "B"],
            attrs_values=[["a1", "a2", "a3"], ["b1", "b2", "b3"]],
            attrs_targets=["dtime", "goal"],
            max_step=10
        ), 
        target_tasks_params_names=["length", "delay_cost"],
        dbg=True)
    
    pack = scheduler.ientry.gen(4)
    print("pack:")
    print(pack)
    scheduler(pack)

    
if __name__ == "__main__":
    test()
