import sys


def parse_cmd_arg(task_name, arg_name, default=None, defaults=[]):
    '''
    - ``default`` -- is not given error will be raised (argument is mandatory)
    otherwise it is optional.
    -``defaults`` -- list of arguments to support.
    '''
    name = arg_name
    if name in sys.argv:
        value = sys.argv[sys.argv.index(name)+1]
        if len(defaults) > 0:
            if value not in defaults:
                raise Exception(
                    "Task: '%s': '%s' argument support only: '%s'"
                    % (task_name, name, str(defaults)))
        print("Task: %s: filename: %s" % (task_name, value))
    else:
        if default is None:
            raise Exception("Task: '%s': '%s' argument needed"
                            % (task_name, name))
        else:
            print("Task: '%s': Warning: argument '%s' not gived, used %s"
                  % (task_name, arg_name, str(default)))
            value = default
    return value
