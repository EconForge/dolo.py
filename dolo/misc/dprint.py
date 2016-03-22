import inspect

class bcolors:

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''


def dprint(s):
    '''Prints `s` with additional debugging informations'''

    import inspect

    frameinfo = inspect.stack()[1]
    callerframe = frameinfo.frame
    d = callerframe.f_locals

    if (isinstance(s,str)):
        val = eval(s, d)
    else:
        val = s
        cc = frameinfo.code_context[0]
        import re
        regex = re.compile("dprint\((.*)\)")
        res = regex.search(cc)
        s = res.group(1)

    text = ''
    text += bcolors.OKBLUE + "At <{}>\n".format(str(frameinfo)) + bcolors.ENDC
    text += bcolors.WARNING + "{}:  ".format(s) + bcolors.ENDC
    text += str(val)
    text += str()

    print(text)

if __name__ == '__main__':

    a = 34


    dprint('a')

    dprint(a)
