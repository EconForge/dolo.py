def portfolios_to_deterministic(model,pf_names):

    #######
    #######

    import re
    regex = re.compile('.*<=(.*)<=.*')
    for i,eq in enumerate(model['equations']):
        from dolo.symbolic.symbolic import Variable, Equation
        if 'complementarity' in eq.tags:
            m = regex.match(eq.tags['complementarity'])
            vs = m.group(1).strip()
            if vs in pf_names:
                v = Variable(vs,0)
                neq = Equation(v,0)
                neq.tag(**eq.tags)
                model['equations'][i] = neq

    print('Warning : initial model changed')

    return model