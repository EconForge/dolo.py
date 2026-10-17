import time
import numpy

try:
    from rich.console import Console

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class IterationsPrinter:
    def __init__(self, *knames, verbose=False, rich=True):
        knames_dict = dict(knames)
        names = []
        types = []
        fmts = []
        for k, v in knames_dict.items():
            names.append(k)
            types.append(v)
            if v == int:
                mm = max([4, len(k)])
                fmts.append("{{:{}}}".format(mm))
            elif v == float:
                fmts.append("{:9.3e}")
        fmt_str = "| " + str.join(" | ", fmts) + " |"

        self.verbose = verbose
        self.names = names
        self.types = types
        self.fmts = fmts
        self.width = len(fmt_str.format(*[0 for _ in self.names]))
        self.fmt_str = fmt_str
        self.t_start = time.time()

        self.use_rich = HAS_RICH and rich and (verbose is not False)
        self.console = Console() if self.use_rich else None

        # Precompute Rich column widths and styles
        if self.use_rich:
            self.col_widths = []
            self.col_styles = []
            for name, typ in zip(self.names, self.types):
                if typ == int:
                    w = max(len(name), 4)
                    style = "cyan"
                elif name in ("Error", "f_x", "d_x", "x_err", "v_err"):
                    w = max(len(name), 9)
                    style = "bold magenta"
                elif name in ("Gain", "Lambda_0"):
                    w = max(len(name), 7)
                    style = "green"
                elif "Time" in name:
                    w = max(len(name), 7)
                    style = "yellow"
                else:
                    w = max(len(name), 4)
                    style = "blue"

                self.col_widths.append(w)
                self.col_styles.append(style)

            # Box borders
            self.box_top = "╭" + "┬".join("─" * (w + 2) for w in self.col_widths) + "╮"
            self.box_sep = "├" + "┼".join("─" * (w + 2) for w in self.col_widths) + "┤"
            self.box_bot = "╰" + "┴".join("─" * (w + 2) for w in self.col_widths) + "╯"
            self.table_width = len(self.box_top)
            self._header_printed = False
            self._closed = False

    def print_line(self):
        if not self.verbose:
            return
        if not self.use_rich:
            print("-" * self.width)

    def print_header(self, msg=None):
        if not self.verbose:
            return
        self.t_start = time.time()

        if self.use_rich:
            if msg:
                self.console.print(f"[bold cyan]{msg:^{self.table_width}}[/bold cyan]")
            self.console.print(self.box_top)
            hdr_cells = [f" {h:>{w}} " for h, w in zip(self.names, self.col_widths)]
            self.console.print("[bold]│" + "│".join(hdr_cells) + "│[/bold]")
            self.console.print(self.box_sep)
            self._header_printed = True
            self._closed = False
        else:
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

        if self.use_rich:
            if not getattr(self, "_header_printed", False):
                self.print_header()

            row_cells = []
            for k, typ, w, style in zip(
                self.names, self.types, self.col_widths, self.col_styles
            ):
                val = args.get(k, None)
                if val is None:
                    txt = " " * w
                elif typ == int:
                    txt = f"{int(val):>{w}}"
                elif typ == float:
                    if numpy.isnan(val):
                        txt = f"{'nan':>{w}}"
                        style = "dim"
                    elif "Time" in k:
                        txt = f"{val:.3f}s"
                        txt = f"{txt:>{w}}"
                    elif k in ("Gain", "Lambda_0"):
                        if abs(val) < 100:
                            txt = f"{val:>{w}.3f}"
                        else:
                            txt = f"{val:>{w}.2e}"
                    else:
                        txt = f"{val:>{w}.3e}"
                else:
                    txt = f"{str(val):>{w}}"

                row_cells.append(f" [{style}]{txt}[/{style}] ")

            self.console.print("│" + "│".join(row_cells) + "│")
        else:
            vals = [args[k] for k in self.names]
            print(self.fmt_str.format(*vals))

    def print_finished(self):
        if not self.verbose:
            return
        elapsed = time.time() - self.t_start

        if self.use_rich:
            if getattr(self, "_header_printed", False) and not getattr(
                self, "_closed", False
            ):
                self.console.print(self.box_bot)
                self._closed = True
            self.console.print(
                f"[dim]Total elapsed: [bold yellow]{elapsed:.2f}s[/bold yellow][/dim]\n"
            )
        else:
            line = "| Elapsed: {:.2f} seconds.".format(elapsed)
            self.print_line()
            print(line + " " * (self.width - len(line) - 1) + "|")
            self.print_line()
            print()
