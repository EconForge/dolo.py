import sys
is_python_3 =  sys.version_info >= (3, 0)

def read_file_or_url(url):

    if 'http' in url:
        if is_python_3:
            import urllib.request
            txt = urllib.request.urlopen(url).read()
            txt = txt.decode('utf8') # not very robust
        else:
            import urllib2
            txt = urllib2.urlopen(url).read()
    else:
        # must be a file
        with open(url,  encoding='utf8') as f:
            txt = f.read()

    return txt



def pcat(filename, target='ipython'):

    code = read_file_or_url(filename)

    HTML_TEMPLATE = """<style>
    {}
    </style>
    {}
    """

    from pygments.lexers import get_lexer_for_filename
    lexer = get_lexer_for_filename(filename, stripall=True)

    from pygments.formatters import HtmlFormatter, TerminalFormatter
    from pygments import highlight

    try:
        assert(target=='ipython')
        from IPython.display import HTML, display
        from pygments.formatters import HtmlFormatter
        formatter = HtmlFormatter(linenos=True, cssclass="source")
        html_code = highlight(code, lexer, formatter)
        css = formatter.get_style_defs()
        html = HTML_TEMPLATE.format(css, html_code)
        htmlres = HTML(html)

        return htmlres

    except Exception as e:
        print(e)
        pass

    formatter = TerminalFormatter()
    output = highlight(code,lexer,formatter)
    print(output)
