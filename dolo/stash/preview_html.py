from dolo.stash.preview_latex import  LatexPrinter

def model_preview(model, output_file, eq_range=None, print_info = True,jsMathPath='jsMath',styleSheet='./style.css',latex_names={}):
    lp = LatexPrinter()

    def latex(expr):
        return lp.doprint(expr)

    if eq_range == None:
        equations = model.equations
    else:
        equations = model.equations[eq_range]
        
    base = '''
    <html>
    <head>
    <title>Equations list</title>

    <script src="%s/easy/load.js"></script>
    <link rel="StyleSheet" href="{styleSheet}" type="text/css" media="screen">


    </head>
    <body>
    <div class="pagetitle">Equations for model : %s</div>
    <div class="section">
        <div class="section_title">Infos</div>
        <div class="section_content">
            %s
        <div>
    </div>
    <div class="section">
        <div class="section_title">Equations</div>
        <div class="section_content">
            %s
        <div>
    </div>
    </body>
    </html>
    '''.format(styleSheet=styleSheet)
    info_html = '''
            <div class="sub_section">
                <div class="sub_section_title">Variables</div>
                <div class="sub_section_content">
                    %s
                </div>
                <div class="sub_section_title">Shocks</div>
                <div class="sub_section_content">
                    %s
                </div>
                <div class="sub_section_title">Parameters</div>
                <div class="sub_section_content">
                    %s
                </div>
            </div>
    '''
    variables_string =  '<div class="math">%s</div>' %str.join(' , ', [ latex(v).strip('$') for v in model.variables ])
    shocks_string = '<div class="math">%s</div>' %str.join(' , ', [ latex(v).strip('$') for v in model.shocks ])
    parameters_string = '<div class="math">{%s}</div>' %str.join(' , ', [ latex(v).strip('$') for v in model.parameters ])
    
    info_html = info_html %( variables_string , shocks_string , parameters_string )
    
    eqs_html = ""
    for eq in equations:
        tex = lp.doprint(eq).strip("$")
        
        # quickfix for jsMath not allowing \operatorname
        tex = tex.replace('\operatorname{log}','\log').replace('\operatorname{exp}','\exp')
        
        if eq.name != None:
            name = eq.name
        else:
            name = "Unlabeled equation"
        title = '(%s) %s' %(eq.n, name)
        htmleq = '''
        <div class="sub_section">
            <div class="sub_section_title">%s</div>
            <div class="sub_section_content">
                <div class="math">%s</div>
            </div>
        </div>
        '''
        eqs_html += htmleq % (title,tex)
    
    output = base % (jsMathPath, model.fname, info_html ,eqs_html)
    
    f = file(output_file,'w')
    f.write(output)
    f.close()
