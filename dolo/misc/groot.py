# I Go to Root.
def groot():
    "Changes current directory to the root of the project (looks for README.md)."

    def check_filelist(l):
        if ('README.md' in l) or ('README' in l):
            return True
        else:
            return False

    import os, copy
    cwd = os.getcwd() # initial dir
    cwd0 = copy.copy(cwd)
    cwd_init = copy.copy(cwd)

    found = False
    sysroot = False

    while not found and not sysroot:
        found = check_filelist(os.listdir())
        if not found:
            os.chdir(os.path.join(cwd, os.pardir))
            cwd = os.getcwd()
            if cwd == cwd0:
                sysroot = True
            else:
                cwd0 = cwd

    if sysroot:
        os.chdir(cwd_init)
        raise Exception("Root directory not found.")
