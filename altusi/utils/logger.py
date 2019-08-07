"""
Logger module
=============
"""


import time
from termcolor import colored


# log mode
NONE = 0
INFO = 1
DEBUG = 2
BEGIN = 4
END = 8
ERROR = 16

LOG_MAP = {
    INFO:   'INFO',
    DEBUG: 'DEBUG',
    BEGIN: 'BEGIN',
    END:   'END',
    ERROR: 'ERROR'
}

# LOG_ENABLE = NONE
# LOG_ENABLE = INFO
# LOG_ENABLE = INFO | ERROR
LOG_ENABLE = INFO | BEGIN | END | DEBUG | ERROR


def LOG(log_mode, log_txt, obj=None):
    """Print log while running program

    Parameters
    ----------
    log_mode : int
        Logging mode
    log_txt : str
        String to log
    obj : Optional[object]
        Extra object to display
    """
    if LOG_ENABLE & log_mode:
        if obj is not None:
            log_str = '{} {}'.format(log_txt, obj)
        else:
            log_str = '{}'.format(log_txt)

        color = None
        if log_mode == INFO:
            color = 'blue'
        elif log_mode == ERROR:
            color = 'red'
        elif log_mode in [BEGIN, END]:
            color = 'white'
        elif log_mode == DEBUG:
            color = 'yellow'

        log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        log_prefix = '[{}]'.format(LOG_MAP[log_mode])
        print('[{}]'.format(colored(log_time, color='green')),
              '{}'.format(colored('{:<7s}'.format(log_prefix),
                                  color=color, attrs=['bold'])),
              log_str)
    else:
        pass
