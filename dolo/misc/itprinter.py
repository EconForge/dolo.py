import time


class IterationsPrinter:
    def __init__(self, *knames, verbose=False):
        knames = dict(knames)
        names = []
        types = []
        fmts = []
        for k, v in knames.items():
            names.append(k)
            types.append(v)
            if v == int:
                mm = max([4, len(k)])
                # print(mm)
                fmts.append("{{:{}}}".format(mm))
            elif v == float:
                fmts.append("{:9.3e}")
        fmt_str = "| " + str.join(" | ", fmts) + " |"
        self.verbose = verbose
        self.names = names
        self.types = types
        self.fmts = fmts
        self.width = len(fmt_str.format(*[0 for i in self.names]))
        self.fmt_str = fmt_str
        self.t_start = time.time()

    def print_line(self):
        if not self.verbose:
            return
        print("-" * self.width)

    def print_header(self, msg=None):
        if not self.verbose:
            return
        self.print_line()
        if msg is not None:
            ll = "| " + msg
            print(ll + " " * (self.width - len(ll) - 1) + "|")
            self.print_line()
        title_str = ""
        for i, v in enumerate(self.types):
            k = self.names[i]
            if v == int:
                title_str += " {:4} |".format(k)
            elif v == float:
                title_str += " {:9} |".format(k)
        title_str = "|" + title_str
        print(title_str)
        self.print_line()

    def print_iteration(self, **args):
        if not self.verbose:
            return
        vals = [args[k] for k in self.names]
        print(self.fmt_str.format(*vals))

    def print_finished(self):
        if not self.verbose:
            return
        elapsed = time.time() - self.t_start
        line = "| Elapsed: {:.2f} seconds.".format(elapsed)
        self.print_line()
        print(line + " " * (self.width - len(line) - 1) + "|")
        self.print_line()
        print()
