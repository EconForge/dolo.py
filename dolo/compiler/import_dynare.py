
def import_dynare(filename, output_type=None):

    import re

    with open(filename) as f:

        txt = f.read()

        txt = re.sub(re.compile("//.*?\n"), "", txt)

        txt = txt.replace("\n", " ")
        txt = txt.replace("\r", " ")
        txt = txt.replace("\t", " ")

        txt = txt.replace(";", ";\n")


        import re
        reg_var = re.compile("var (.*?);")
        resp = reg_var.findall(txt)
        variables = [v.strip() for v in str.split(resp[0].replace(',',' '), " ")]
        variables = [v for v in variables if len(v) > 0]

        reg_varexo = re.compile("varexo (.*?);")
        resp = reg_varexo.findall(txt)
        shocks = [v.strip() for v in str.split(resp[0].replace(',',' '), " ")]
        shocks = [v for v in shocks if len(v) > 0]

        reg_params = re.compile("parameters (.*?);")
        resp = reg_params.findall(txt)
        parameters = [v.strip() for v in str.split(resp[0].replace(',',' '), " ")]
        parameters = [v for v in parameters if len(v) > 0]

        reg_paraminit = re.compile("parameters(.*?);(.*)model;", re.MULTILINE + re.DOTALL)
        resp = reg_paraminit.findall(txt)[0][1]
        resp = resp.replace("\n", '')
        pinit = str.split(resp, ';')
        pinit = [[h.strip() for h in str.split(e, "=")] for e in pinit]
        pinit = [e for e in pinit if len(e) == 2]

        reg_init = re.compile("initval;(.*?)end;", re.MULTILINE + re.DOTALL)
        resp = reg_init.findall(txt)[0]
        resp = resp.replace("\n", '')
        init = str.split(resp, ';')
        init = [[h.strip() for h in str.split(e, "=")] for e in init]
        init = [e for e in init if len(e) == 2]

        calib = {}
        for k, v in pinit:
            calib[k] = v.replace("^", '**')
        for k, v in init:
            calib[k] = v.replace("^", '**')

        equations_block = re.compile("model;(.*?)end;", re.MULTILINE + re.DOTALL).findall(txt)[0]
        eqs = str.split(equations_block, ';')[:-1]
        regex_eq = re.compile('.*?\[(.*)\](.*)|(.*)')
        eqs = [eq.strip() for eq in eqs]
        matches = [regex_eq.match(eq) for eq in eqs]
        eqs = [m.group((2)) if m.group(1) else m.group(3) for m in matches]
        eqs = [str.strip(s).replace("^", '**') for s in eqs if str.strip(s) != '']


        from collections import OrderedDict

        fname = "ABKM2"
        model_name = 'ABKM2'
        model_type = 'dynare'

        infos = dict()
        infos['filename'] = fname
        infos['name'] = model_name
        infos['type'] = model_type
        symbols = OrderedDict(variables=variables, shocks=shocks, parameters=parameters)
        symbolic_equations = eqs
        symbolic_calibration = calib

        for v in variables + shocks + parameters:
            if v not in symbolic_calibration:
                symbolic_calibration[v] = 0

        if output_type == 'json':
            d = {
                'name': model_name,
                'type': model_type,
                'symbols': symbols,
                'equations': symbolic_equations,
                'calibration': symbolic_calibration
            }

            return d

        from dolo.compiler.model_symbolic import SymbolicModel
        options = {}
        smodel = SymbolicModel(model_name, model_type, symbols, symbolic_equations,
                               symbolic_calibration,
                               options=options, definitions=None)

        from dolo.compiler.model_dynare import DynareModel
        model = DynareModel(smodel, infos=infos, options={})
        return model
