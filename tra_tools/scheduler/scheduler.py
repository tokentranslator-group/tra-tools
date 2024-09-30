# $ python3 -m tra_tools.scheduler.schedule
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

    def estimating_from(self, pack):
        '''Given the pack which have target_attrs, use them
        to estimate `self.gmap_attrs_t_n_st` accumulatively
        i.e. with use of previous runs.
        '''
        with self:
            df = self.load(pack)
            self.group_by_named(df)

    def apply_heuristic_to(self, pack):
        '''Apply estimated heuristics to given pack.'''
        assert hasattr(self, 'gmap_attrs_t_n_st')

        with self:
            df = self.load(pack)
            self.mk_heuristics(df)
        return self.heuristics

    def __call__(self, pack, cpu_count=8, solve=True):
        '''self.gmap_attrs_t_n_st will be preserved in each call allowing
        for the heuristics to be accumulative'''

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

            if solve:
                if self.dbg:
                    print("\n================ running: scheduler.solve():")
                self.solve()

                if self.dbg:
                    self.plot()

    def __enter__(self):
        '''Clear the `self` state'''

        # this do not need to be cleared since
        # it will be accumulative:
        # if hasattr(self, 'gmap_attrs_t_n_st'):
        #     del self.gmap_attrs_t_n_st

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
        '''Group by each named attribute and having calculated
        the targets ones store them in a `self.gmap_attrs_t_n_st` dict.
        If `self.gmap_attrs_t_n_st` alredy exist it will be added to
        so there is some accumulative effect.
        
        '''
        named_attrs = self.columns
        # named_attrs = columns[:-4]
        
        if hasattr(self, 'gmap_attrs_t_n_st'):
            if self.dbg:
                print("using previous heuristics as the init one!")
            gmap_attrs_t_n_st = self.gmap_attrs_t_n_st
        else:
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
            if self.dbg:
                print("nattr_classes.sum:")
                print(nattr_classes)
            
            for tattr_name in target_attrs_names: 

                # then project each target_attribute of factorized table:
                # (i.e for each value of `nattr_name` (i.e. `nattr_value`)
                # for each `tattr_name` find the `tattr`s summed value):
                # (ex: for these like ("nattr_value", "sum of tattr_value"))
                nattr_values = list(nattr_classes.index.array)
                table_nval_stval = dict(zip(nattr_values,
                                            list(nattr_classes[tattr_name])))
                if self.dbg:
                    print("tattr_name:", tattr_name)
                    print("nattr_name:", nattr_name)
                    print("table_nval_stval:")
                    print(table_nval_stval)
                    
                # and map resulting table from tattr to nattr
                gmap_attrs_t_n_st = self._update_gmap(
                    gmap_attrs_t_n_st,
                    nattr_classes, table_nval_stval,
                    nattr_name, tattr_name)

        if self.dbg:
            print("\ngmap_attrs_t_n_st:")
            print(gmap_attrs_t_n_st)

        self.gmap_attrs_t_n_st = gmap_attrs_t_n_st
        
    def _update_gmap(self, gmap,
                     nattr_classes, table_nval_stval,
                     nattr_name, tattr_name):
        ''' To map the direct product of `table_nval_stval` table
        to directional: from `tattr` to `nattr`
        and if there is a value (from previous `group_by_named` calls)
        then update them.
        '''
        # ['b1', 'b2', 'b3', ...]
        for nattr_value in list(nattr_classes.index.array):
            # if the value is alredy present
            # from the previous packs:
            # if not it will just be empty dict for each tattr_name:
            # like {dtime:{}, goal:{}}
            if nattr_name in gmap[tattr_name]:
                if nattr_value in gmap[tattr_name][nattr_name]:
                    gmap[tattr_name][nattr_name][nattr_value]\
                        += table_nval_stval[nattr_value]
                # print("added")
                # print(gmap[tattr_name][nattr_name])
                # print(nattr_value)
                # print(list(nattr_classes.index.array))
            else:
                gmap[tattr_name][nattr_name]\
                    = table_nval_stval

        return gmap

    def mk_heuristics(self, df):
        '''
        For estimating each row in the table
        according to the attrs values found in `self.group_by_named`:
        - ``df`` -- the DataFrame to which  the `gmap_attrs_t_n_st`
        been applied.

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

            # inner mapper of columns:
            def mapper1(cNmVal):
                n_st = gmap_attrs_t_n_st[param][cNmVal[0]]
                if cNmVal[1] in n_st:
                    return n_st[cNmVal[1]]
                else:
                    return 0

            row1 = list(map(
                # columnNameVal is name/value
                # for each attribute (nattr) in the row:
                mapper1,
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
            heuristics[tattr_name] = param_expend
        self._heuristics = heuristics

        heuristics_norm = {}
        for tattr_name in heuristics:
            param_expend = heuristics[tattr_name]
            param_expend1 = self._normalize_heuristic(tattr_name, param_expend)

            heuristics_norm[tattr_name] = param_expend1
        self.heuristics = heuristics_norm

        if self.dbg:
            print("heuristics:")
            print(self.heuristics)

    def _normalize_heuristic(
            self, tattr_name, param_expend, invert_goal=False):
        '''
        - ``invert_goal``-- use `True` for solving scheduler
        since than quicker then better but for the heuristics use `False`
        (as `dtime` one)'''

        # convert to int:
        param_expend1 = (param_expend/param_expend.sum()*100).astype(int)
        if tattr_name == "goal" and invert_goal:
            # invert: than less the goal is that better will the result be
            # and hence the dalay_cost must be bigger.
            param_expend1 = param_expend1.max()-param_expend1

        if self.dbg:
            print("\ntattr_name:", tattr_name)
            print("param_expend:")
            print(param_expend1)

        return param_expend1

    def has_heuristic(self):
        return hasattr(self, 'gmap_attrs_t_n_st')
        # return hasattr(self, 'heuristics')

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


def test2():
    '''Test heuristics'''
    scheduler = init_sceduler(dbg=False)
    pack = scheduler.ientry.gen(3)
    print("pack to estimate:")
    print(scheduler.ientry.to_pandas(pack))
    scheduler.estimating_from(pack)
    print("estimated gmap_attrs_t_n_st:")
    print(scheduler.gmap_attrs_t_n_st)
    
    pack1 = scheduler.ientry.gen(7)
    print("pack1 to apply heuristics:")
    print(scheduler.ientry.to_pandas(pack1))
    
    nheuristics = scheduler.apply_heuristic_to(pack1)
    heuristics = scheduler._heuristics
    print("estimated heuristic:")
    print(heuristics)

    print("\nestimated heuristic norm:")
    print(nheuristics)


def test_heuristic_unormal():
    scheduler = init_sceduler(dbg=False)
    pack = scheduler.ientry.gen(3)
    scheduler.estimating_from(pack)
    named_attrs = scheduler.columns
    pack = scheduler.ientry.to_pandas(pack)
    print(pack)

    pack_named_only = pack.loc[:, named_attrs]
    gmap = scheduler.gmap_attrs_t_n_st
    heuristics = {}
    for tname in scheduler.columns_targets:
        heuristics[tname] = []
        for entry in pack_named_only.to_numpy():
            res = 0
            nvalues = entry
            # print("entry nvalues:", nvalues)
            for idx, nattr in enumerate(named_attrs):
                nvalue = nvalues[idx]
                n_st = gmap[tname][nattr]
                if nvalue in n_st:
                    res += n_st[nvalue]
            heuristics[tname].append(res)
    print("heuristics0")
    print(heuristics)
    pack1 = scheduler.ientry.to_entries(pack)

    scheduler.apply_heuristic_to(pack1)
    # unnormalized:
    heuristics1 = scheduler._heuristics
    print("heuristics1:")
    print(heuristics1)
    for tname in heuristics:
        assert np.all(heuristics[tname] == heuristics1[tname])
    print("test succ")


def test1(n=3):
    '''Testing `schduler.gmap_attrs_t_n_st`
    accumulativeness.'''

    scheduler = init_sceduler()
    for idx in range(n):
        print("############# step %d ###########" % idx)
        pack = scheduler.ientry.gen(4)
        scheduler(pack, solve=False)
        with open("/tmp/tmp.txt", "a") as f:
            f.write(str(scheduler.gmap_attrs_t_n_st)+"\n")


def test(cpu_count=2):
    '''Basic test'''
    scheduler = init_sceduler()

    pack = scheduler.ientry.gen(4, fill_random=False)
    print("pack:")
    print(pack)
    scheduler(pack, cpu_count=cpu_count)


def init_sceduler(dbg=True):
    from tra_tools.zipper import Entry

    scheduler = CpuScheduler(
        init_entry=Entry(
            attrs_names=["A", "B"],
            attrs_values=[["a1", "a2", "a3"], ["b1", "b2", "b3"]],
            attrs_targets=["dtime", "goal"],
            max_step=10
        ), 
        target_tasks_params_names=["length", "delay_cost"],
        dbg=dbg)
    
    return scheduler
    

if __name__ == "__main__":
    test2()
    # test_heuristic_unormal()
    # test1()
    # test()
