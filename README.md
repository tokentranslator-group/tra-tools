# tra-tools
Some useful tools.

### Requirements
```
pip install -r requirements.txt
```

### Installation and running
```
pip install tratools
```

### Usage:

1. Progress:
```
>>> import tratools.progresses.progress_cmd as progress_cmd
>>> ProgressCmd = progress_cmd.ProgressCmd
>>> progress = ProgressCmd(steps, prefix="loading:")
>>> progress.succ(idx)  # somewhere in loop
>>> progress.print_end()
```

2. Zipper:
```
>>> from tratools.zipper import Zipper, Entry
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

>>> pack, pack_size = zipper.next()
>>> pack = list(pack)
>>> print("pack:")
>>> for e in pack:
>>>     print(e.attrs_values_entry)

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

>>> zipper.update(pack)
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

```

3. CpuScheduler:
```
    >>> from tratools.scheduler import CpuScheduler
    >>> from tratools.zipper import Entry

    >>> scheduler = CpuScheduler(
    >>>    init_entry=Entry(
    >>>        attrs_names=["A", "B"],
    >>>        attrs_values=[["a1", "a2", "a3"], ["b1", "b2", "b3"]],
    >>>        attrs_targets=["dtime", "goal"],
    >>>        max_step=10
    >>>    ), 
    >>>    target_tasks_params_names=["length", "delay_cost"],
    >>>    dbg=True)
    
    >>> pack = scheduler.ientry.gen(4)
    >>> scheduler(pack)

    solution:
    (1_len_27_delay_4_0, cpu4, 0, 27)
    (2_len_19_delay_7_0, cpu1, 0, 19)
    (3_len_19_delay_7_0, cpu0, 0, 19)
    (0_len_33_delay_0_0, cpu5, 11, 44)

    # see `tratools.scheduler.test()` for details.
```