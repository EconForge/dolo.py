import sys

is_python_3 = sys.version_info >= (3, 0)


def read_file_or_url(url):

    if "http" in url:
        if is_python_3:
            import urllib.request

            txt = urllib.request.urlopen(url).read()
            txt = txt.decode("utf8")  # not very robust
        else:
            import urllib2

            txt = urllib2.urlopen(url).read()
    else:
        # must be a file
        with open(url, encoding="utf8") as f:
            txt = f.read()

    return txt


def pcat(filename, target="ipython"):

    code = read_file_or_url(filename)

    try:
        assert target == "ipython"
        from IPython import get_ipython

        ipython = get_ipython()

        ipython.magic(f"pycat {filename}")
        return

    except Exception as e:
        print(e)
        pass

    from pygments.lexers import get_lexer_for_filename
    from pygments.formatters import TerminalFormatter
    from pygments import highlight

    lexer = get_lexer_for_filename(filename, stripall=True)
    formatter = TerminalFormatter()
    output = highlight(code, lexer, formatter)
    print(output)
