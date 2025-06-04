import sys


def parse_cmd_arg(task_name, arg_name, default=None, defaults=[]):
    '''
    - ``default`` -- is not given error will be raised (argument is mandatory)
    otherwise it is optional and will return it in case no been found in args.

    -``defaults`` -- list of arguments to support.

    Ex:

    >>> from tra_tools.io import parse_cmd_arg
    >>> todo = parse_cmd_arg("global", '-run')
    >>> mode = parse_cmd_arg(task_name, '-mode', defaults=['query', 'mk_notes', 'mk_goals'])
    >>> replace = eval(parse_cmd_arg(task_name, '-replace', default="False"))
    '''
    name = arg_name
    if name in sys.argv:
        value = sys.argv[sys.argv.index(name)+1]
        if len(defaults) > 0:
            if value not in defaults:
                raise Exception(
                    "Task: '%s': '%s' argument support only: '%s'"
                    % (task_name, name, str(defaults)))
        print("Task: %s: %s: %s" % (task_name, name, value))
    else:
        if default is None:
            raise Exception("Task: '%s': '%s' argument needed"
                            % (task_name, name))
        else:
            print("Task: '%s': Warning: argument '%s' not gived, used %s"
                  % (task_name, arg_name, str(default)))
            value = default
    return value
