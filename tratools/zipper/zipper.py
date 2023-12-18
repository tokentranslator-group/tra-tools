import os
import numpy as np
import pandas as pd
import pickle 
from functools import reduce
import itertools as it


class Entry():
    '''Series type with succ'''

    # the type props:
    attrs_names = []
    attrs_values = []
    attrs_targets = []
    
    def __init__(self,
                 attrs_names=None,
                 attrs_values=None,
                 attrs_targets=None,
                 
                 attrs_values_entry=None,
                 max_step=-1):
        '''If some of the input values will not be given
        then the according properties will not been updated '''

        # FOR the class/type props:
        if attrs_names is not None:
            self.attrs_names.clear()
            self.attrs_names.extend(attrs_names)
        if attrs_values is not None:
            self.attrs_values.clear()
            self.attrs_values.extend(attrs_values)
        if attrs_targets is not None:
            self.attrs_targets.clear()
            self.attrs_targets.extend(attrs_targets)

        self.param_names = []

        self.attrs_values_entry = attrs_values_entry
        self.max_step = max_step

    def gen(self, n, fill_targets=True):
        '''
        - ``fill_targets`` -- if true then assume the targets attributes values
        have not been given and generate them randomly'''

        entries = list(self.to_entries(self.gen_states(count=n)))
        if fill_targets:
            for entry in entries:
                attrs_targets = entry.attrs_targets
                attrs_gen_values = np.random.uniform(0, 1, len(attrs_targets))
                entry.update(attrs_targets, attrs_gen_values)

                # entry.attrs_values_entry.loc[:, attrs_targets] = attrs_gen_values
        return entries

    def to_pandas(self, entries):
        '''
            # for convertin DataFrame to entries and back:
            entries = list(self.ientry.to_entries(df))
            df1 = self.ientry.to_pandas(entries)

        '''
        df = self.init_df()
        for entry in entries:
            # ignore_index to generate new index for anew entry: 
            df = pd.concat([df, entry.attrs_values_entry], ignore_index=True)
            # df = df.append(entry.attrs_values_entry, ignore_index=True)
        return df

    def init_df(self):
        columns = self.attrs_names + self.param_names
        columns_targets = self.attrs_targets
        return init_df(columns, columns_targets)

    def to_args(self):
        # drop what is NaNs:
        df = self.attrs_values_entry.dropna(axis="columns", how="all")
        # return first and only entry
        return list(df.to_numpy()[0])

    def succ(self, attrs_values_entry):
        '''Will produce another instance with the same
        properties.
        Entry should be the Series, otherwise 
        the values in attrs_values_entry (list like)
        should match following attributes of self:
        self.attrs_names+self.param_names+self.attrs_targets'''

        # print(type(attrs_values_entry))
        ''' TODO: dropna(all) in to_args also should be considered
        for that case:
        if type(attrs_values_entry) == pd.Series:
            new_attrs_values_entry = attrs_values_entry.copy()
            # print("new_attrs_values_entry:")
            # print(new_attrs_values_entry)
        else:
        '''
        df = self.init_df()
        columns = self.attrs_names + self.param_names
        columns.extend(self.attrs_targets)
        # print("columns")
        # print(columns)
        # print("attrs_values_entry")
        # print(attrs_values_entry)
        d = pd.DataFrame(dict(zip(
            columns,
            map(lambda x: [x], attrs_values_entry))))

        new_attrs_values_entry = pd.concat(
            [df, d],
            ignore_index=True)
        '''
        new_attrs_values_entry = df.append(dict(zip(
            columns, attrs_values_entry)), ignore_index=True)
        '''
        # print(new_attrs_values_entry)
        return(self.__class__(attrs_values_entry=new_attrs_values_entry))

    def to_entries(self, table):
        '''mapping of table of values to self.__class__
        type.
        '''
        if type(table) == pd.DataFrame:
            index = list(table.index.array)
            table = [table.loc[idx] for idx in index]
        
        return(map(self.succ, table))
    
    def gen_states(self, count=None):
        if count is None:
            max_step = self.max_step if self.max_step > 0 else np.inf
        else:
            max_step = count

        states = list(map(
            lambda idxElm: idxElm[1],
            it.takewhile(
                lambda idxElm: idxElm[0] < max_step,
                enumerate(it.product(*self.attrs_values)))))
        return states

    def update(self, names, values):
        self.attrs_values_entry.loc[:, names] = values


class Zipper():
    '''Store todos and dones tables and provide a way to take packs
    of values from the former (with the `self.next` func)
    and give one to the later (with the `self.update func`).
    The type of stored values is pandas.DataFrame, but converted
    with `init_entry.to_entries` function in the `self.next`. 

    - ``init_entry`` -- is used to convert the current tasks pack to
    an Entry type (which is done by `init_entry.to_entries` func).
    This is also used to take attrs_names and param_names for the columns
    of todos and dones table

    Examples::
        >>> zipper = Zipper(
                init_entry=Entry(
                    attrs_names=["A", "B"],
                    attrs_values=[["a1", "a2", "a3"], ["b1", "b2", "b3"]],
                    attrs_targets=["dtime", "goal"],
                    max_step=10
                ), pack_size=2)
        >>> zipper.load(use_backup=False)
        >>> print("todos:")
        >>> print(zipper.todos)
            A   B
        0  a1  b1
        1  a1  b2
        2  a1  b3
        3  a2  b1
        4  a2  b2
        5  a2  b3
        6  a3  b1
        7  a3  b2
        8  a3  b3

        >>> print("dones:")
        >>> print(zipper.dones)
        Empty DataFrame
        Columns: [A, B]
        Index: []

        >>> pack, pack_size = zipper.next()
        >>> print("pack:")
        >>> for e in pack:
        >>>    print(e.attrs_values_entry)
        A    a1
        B    b1
        Name: 0, dtype: object
        A    a1
        B    b2
        Name: 1, dtype: object

        >>> print("\ntodos after pack:")
        >>> print(zipper.todos)
            A   B
        2  a1  b3
        3  a2  b1
        4  a2  b2
        5  a2  b3
        6  a3  b1
        7  a3  b2
        8  a3  b3

        >>> zipper.update(list(pack))
        >>> print("\ntodos after update:")
        >>> print(zipper.todos)
            A   B
        2  a1  b3
        3  a2  b1
        4  a2  b2
        5  a2  b3
        6  a3  b1
        7  a3  b2
        8  a3  b3

        >>> print("\ndones after update")
        >>> print(zipper.dones)
            A   B
        0  a1  b1
        1  a1  b2
    '''

    def __init__(self, init_entry=None, pack_size=10,
                 dbg=False, use_backup=True):
        self.dbg = dbg
        self.use_backup = use_backup

        self.folder = "result"
        self.ientry = init_entry
        self.columns = self.ientry.attrs_names + self.ientry.param_names
        self.columns_targets = self.ientry.attrs_targets

        self.pack_size = pack_size

        self.register_init_gen(self.ientry.gen_states)

    def load(self, use_backup=True):
        '''Load todos from a backup from self.folder/"zipper_todos.backup" func
        if it exist or init from `self.init_gen` if not.
        Init dones as empty DataFrame'''

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        ftodos = os.path.join(self.folder, "zipper_todos.backup")
        if os.path.exists(ftodos):
            with open(ftodos, "rb") as f:
                self.todos = pickle.load(f)
                print("zipper.todos loaded from ", ftodos)

            fdones = os.path.join(self.folder, "zipper_dones.backup")
            if os.path.exists(fdones):
                with open(fdones, "rb") as f:
                    self.dones = pickle.load(f)
                print("zipper.dones loaded from ", fdones)
        else:
            # init dones:
            self.dones = self.init_df()
            
            # init todos:
            states = self.init_gen()
            # print(states)
            df_todos = self.init_df()
            self.todos = (reduce(
                lambda acc, state: pd.concat(
                    [acc, pd.DataFrame(dict(zip(self.columns, map(lambda x: [x], state))))],
                    
                    # to append with anew generated index:
                    ignore_index=True),
                states, df_todos))
            '''
            self.todos = (reduce(
                lambda acc, state: acc.append(
                    dict(zip(self.columns, state)),

                    # to append with anew generated index:
                    ignore_index=True),
                states, df_todos))
            '''
        if self.dbg:
            print("todos:")
            print(self.todos)
        return len(self.todos)

    def init_df(self):
        return init_df(self.columns, self.columns_targets)

    def next(self):
        '''take pack with the size `self.pack_size` from todos,
        converting it with the self.ientry.to_entries funct before returning'''
        
        # take a pack:
        self.current = self.todos.iloc[:self.pack_size]
        # self.current = self.todos.head(self.pack_size)
        
        # drop columns that is all NaNs:
        # .dropna(axis="columns", how="all")

        # self.current = self.todos.loc[0:self.pack_size]
        
        # remove the taking pack from todos:
        self.todos = self.todos.iloc[self.pack_size:]
        # self.todos.drop(range(0, self.pack_size), inplace=True, axis="index")
        # print("current:")
        # print(self.current)

        # convert to desired format:
        index = list(self.current.index.array)
        return(self.ientry.to_entries(self.current), len(index))

        # index = list(self.current.index.array)
        # return(self.ientry.to_entries(
        #     [self.current.loc[idx] for idx in index]),
        #        len(index))
        # return self.current.to_numpy()

    def update(self, entries, k=0):
        '''Append all entries to the dones recursively.
        Return amount of appended.'''
        pack = self.ientry.to_pandas(entries)
        
        self.dones = pd.concat([self.dones, pack], ignore_index=True)
        # self.dones = self.dones.append(pack, ignore_index=True)
        if self.use_backup:
            self.backup()

        return len(pack)
        '''
        if len(entries) == 0:
            return k
        first = entries.pop(0)
        # print("entry to append:")
        # print(first.attrs_values_entry)

        self.dones = self.dones.append(first.attrs_values_entry)
        if self.use_backup:
            self.backup()

        del first
        return(self.update(entries, k=k+1))
        '''
    def backup(self):
        # backupping todos:
        path = os.path.join(
            self.folder, "zipper_todos.backup")        
        self.todos.to_pickle(path)
        # if self.dbg:
        print("backuped todos to ", path)

        # for backup dones:
        path = os.path.join(
            self.folder, "zipper_dones.backup")        
        self.dones.to_pickle(path)
        # if self.dbg:
        print("backuped dones to ", path)

        # for saving dones as results:
        path = os.path.join(
            self.folder, "zipper_dones.results")
        with open(path, "w") as f:
                f.write(self.dones.to_csv())
        # if self.dbg:
        print("dones as results saved to ", path)
    
    def register_init_gen(self, f):
        self.init_gen = f


def init_df(columns, columns_targets):
    '''For initiate both todos and dones'''
    return (pd.DataFrame(columns=columns+columns_targets)
            .astype(dict(zip(
                columns_targets,  # dtime, goal
                [np.float]*len(columns_targets)))))
       

def split(length, batch_size, start=0):
    '''Split the indexes [1, ..., length]
    to batches with the size batch_size'''

    batch_count = length // batch_size
    # print("batch_count:", batch_count)

    max_length = batch_count * batch_size
    batchs_idxs = np.array_split(np.arange(start, max_length), batch_count)

    return (batch_count, batchs_idxs)


def test_zipper0():
    
    zipper = Zipper(
        init_entry=Entry(
            attrs_names=["A", "B"],
            attrs_values=[["a1", "a2", "a3"], ["b1", "b2", "b3"]],
            attrs_targets=["dtime", "goal"],
            max_step=10
        ), pack_size=2)
    zipper.load(use_backup=False)
    print("todos:")
    print(zipper.todos)
    print("dones:")
    print(zipper.dones)

    pack, pack_size = zipper.next()
    pack = list(pack)
    print("pack:")
    for e in pack:
        print(e.attrs_values_entry)
    
    print(zipper.ientry.to_pandas(pack))
    print("\ntodos after pack:")
    print(zipper.todos)

    zipper.update(pack)
    print("\ntodos after update:")
    print(zipper.todos)
    print("\ndones after update")
    print(zipper.dones)


if __name__ == "__main__":
    test_zipper0()
