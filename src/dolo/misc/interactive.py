from dolo.symbolic.symbolic import *
from dolo.symbolic.symbolic_interactive import *
from dolo.symbolic.model import *

from sympy import exp,log,sin,cos,tan,Matrix,zeros,pi,atan,sqrt
#from solver.solver import *
import re
import inspect


def parse_dynare_text(txt,add_model=True,full_output=False,names_dict = {}):
    '''
    Imports the content of a modfile into the current interpreter scope
    '''
    # here we call "instruction group", a string finishing by a semicolon
    # an "instruction group" can have several lines
    # a line can be
    # - a comment //...
    # - an old-style tag //$...
    # - a new-style tag [key1='value1',..]
    # - macro-instruction @#...
    # A Modfile contains several blocks (in this order) :
    # - an initblock defining variables, exovariables, parameters, initialization
    #   inside the initblock the order of declaration doesn't matter
    # - a model block with two special lines (model; end;)
    # - optional blocks (like endval, shocks) 
    #    seperated by free matlab instructions in any order;
    # - all other instructions are ignored

    otxt = txt
    otxt = otxt.replace("\r\n","\n")
    otxt = otxt.replace("^","**")

    # first, we remove end-of-line comments : they are definitely lost
    regex = re.compile("(.+)//[^#](.*)")
    def remove_end_comment(line):
        res = regex.search(line)
        if res:
            l = res.groups(1)[0]
            return(l)
        else:
            return line
    txt = str.join("\n",map(remove_end_comment,otxt.split("\n")))

    name_regex = re.compile("//\s*fname\s*=\s*'(.*)'")
    m = name_regex.search(txt)
    if m:
        fname = m.group(1)
    else:
        fname = 'anonymous'
    #txt = otxt

    # name : equation1 #
    #print(txt)
    #exit()
    #line_regex = re.compile("(//#|//|@#|)(.*)")
    line_regex = re.compile(
        "(\s*)$|"
        +"\s*//#(.*)$|"
        +"\s*@#(.*)$|"
        +"\s*//[^#](.*)$|"
        +"\s*\[(.*)\]\s*$|"
        +"(\s*[^\s].*)$"
    )
    tag_regex = re.compile("\s*(\w+)\s*=\s*'(.*)'")




    class Instruction_group():
        def __init__(self,s):
            self.string = s
            self.span = 1 + s.count("\n")
            self.tags = {}
            self.instruction = ""
            self.process_lines()
            return None
        def __repr__(self):
            return str([self.instruction,self.tags])
        def process_lines(self):
            lines = []
            entire_instruction = ""
            for l in self.string.split("\n"):
                g = line_regex.match(l).groups()
                #print(g.groups())
                matches = [i for i in range(len(g)) if g[i]!=None]
                if len(matches) !=1 :
                    raise Exception( "Parsing error" )
                else:
                    i = matches[0]
                if i == 0:
                    # this line is blank, just ignore it
                    lines.append(["blank",g[i]])
                if i == 1:
                    # this line contains tag, read them immediately
                    lines.append(["tags",g[i]])
                    keys_tags = [kt.split(":") for kt in g[i].split('#')]
                    for kt in keys_tags:
                        if len(kt) > 1:
                            self.tags[kt[0].strip()] = kt[1].strip()
                if i == 2:
                    # this line contains macro instructions
                    lines.append(["macro",g[i]])
                if i == 3:
                    # there are regular comments on this line
                    lines.append(["comment",g[i]])
                if i == 4:
                    # new style tags
                    lines.append(["tags",g[i]])
                    pairs = [tag_regex.match(kt).groups() for kt in g[i].split(',')]
                    for p in pairs:
                        if len(p)>1:
                            self.tags[p[0]] = p[1].strip()
                if i == 5:
                    entire_instruction += (" " + g[i])
                    lines.append(("instruction",g[i]))                    
            self.instruction = entire_instruction.strip()
            self.lines = lines

    instruction_groups = [Instruction_group(s) for s in txt.split(";")]

    instructions = [ig.instruction for ig in instruction_groups]
    # currently, we don't check that the modfile is valid

    blocks = []

    imodel = [re.compile('model(\(.*\)|)').match(e)!=None for e in instructions]
    imodel = imodel.index(True)
    #imodel = instructions.index("model") #this doesn't work for "MODEL"

    iend = instructions.index("end")
    init_block = instruction_groups[0:imodel]
    model_block = instruction_groups[imodel:(iend+1)]
    next_instructions = instructions[(iend+1):]
    next_instruction_groups = instruction_groups[(iend+1):]
    
    if 'initval' in next_instructions:
        iinitval = next_instructions.index('initval')
        iend = next_instructions.index('end',iinitval)
        matlab_block_1 = next_instruction_groups[0:iinitval]
        initval_block = next_instruction_groups[iinitval:(iend+1)]
        next_instruction_groups = next_instruction_groups[(iend+1):]
        next_instructions = next_instructions[(iend+1):]
    else:
        initval_block = None
        matlab_block_1 = None
        
    if 'endval' in next_instructions:
        iendval = next_instructions.index('endval')
        iend = next_instructions.index('end',iendval)
        matlab_block_2 = next_instruction_groups[0:iendval]
        endval_block = next_instruction_groups[iendval:(iend+1)]
        next_instruction_groups = next_instruction_groups[(iend+1):]        
        next_instructions = next_instructions[(iend+1):]
    else:
        endval_block = None
        matlab_block_2 = None

    # TODO : currently shocks block needs to follow initval, this restriction should be removed
    if 'shocks' in next_instructions:
        ishocks = next_instructions.index('shocks')
        iend = next_instructions.index('end',ishocks)
        matlab_block_3 = next_instruction_groups[0:ishocks]
        shocks_block = next_instruction_groups[ishocks:(iend+1)]
        next_instruction_groups = next_instruction_groups[(iend+1):]
        next_instructions = next_instructions[(iend+1):]
    else:
        shocks_block = None
        matlab_block_3 = None
    

    init_regex = re.compile("(parameters |var |varexo |)(.*)")
    var_names = []
    varexo_names = []
    parameters_names = []
    declarations = {}
    for ig in init_block:
        if ig.instruction != '':
            m = init_regex.match(ig.instruction)
            if not m:
                raise Exception("Unexpected instruction in init block : " + str(ig.instruction))
            if m.group(1) == '':
                [lhs,rhs] = m.group(2).split("=")
                lhs = lhs.strip()
                rhs = rhs.strip()
                declarations[lhs] = rhs
            else:
                arg = m.group(2).replace(","," ")
                names = [vn.strip() for vn in arg.split()]
                if m.group(1).strip() == 'var':
                    dest = var_names
                elif m.group(1).strip() == 'varexo':
                    dest = varexo_names
                elif m.group(1).strip() == 'parameters':
                    dest = parameters_names
                for n in names:
                    if not n in dest:
                        dest.append(n)
                    else:
                        raise Exception("symbol %s has already been defined".format(n))

    # the following instruction set the variables "variables","shocks","parameters"


    variables = []
    for vn in var_names:
        if vn in names_dict:
            latex_name = names_dict[vn]
        else:
            latex_name = None
        v = Variable(vn,0)
        variables.append(v)

    shocks = []
    for vn in varexo_names:
        if vn in names_dict:
            latex_name = names_dict[vn]
        else:
            latex_name = None
        s = Shock(vn,0)
        shocks.append(s)

    parameters = []
    for vn in parameters_names:
        if vn in names_dict:
            latex_name = names_dict[vn]
        else:
            latex_name = None
        p = Parameter(vn)
        parameters.append(p)


    parse_dict = dict()
    for v in variables + shocks + parameters:
        parse_dict[v.name] = v

    special_symbols = [sympy.exp,sympy.log,sympy.sin]
    for s in special_symbols:
        parse_dict[str(s)] = s

    #frame = inspect.currentframe()
    #for s in variables + shocks + parameters:
    #    frame.f_globals[s.name] = s


    #set_variables(var_names,names_dict=names_dict)
    #set_shocks(varexo_names,names_dict=names_dict)
    #set_parameters(parameters_names,names_dict=names_dict)

    parameters_values = {}
    for p in declarations:
        parameters_values[eval(p,parse_dict)] = eval(declarations[p], parse_dict)

        
    special_symbols = [sympy.exp,sympy.log,sympy.sin,sympy.cos, sympy.atan, sympy.tan]
    for s in special_symbols:
        parse_dict[str(s)] = s

    # Now we read the model block
    model_tags = model_block[0].tags
    equations = []
    for ig in model_block[1:-1]:
        if ig.instruction != '':
            teq = ig.instruction.replace('^',"**")
            teqlhs,teqrhs = teq.split("=")
            eqlhs = eval(teqlhs, parse_dict)
            eqrhs = eval(teqrhs, parse_dict)
            eq = Equation(eqlhs,eqrhs)
            eq.tags.update(ig.tags)
    #        if eq.tags.has_key('name'):
    #            eq.tags[] = ig.tags['name']
            equations.append(eq)
        
    # Now we read the initval block
    init_values = {}
    if initval_block != None:
        for ig in initval_block[1:-1]:
            [lhs,rhs] = ig.instruction.split("=")
            init_values[eval(lhs,parse_dict)] = eval(rhs,parse_dict)
    
    # Now we read the endval block
    # I don't really care about the endval block !
    
    end_values = {}
    if endval_block != None:
        for ig in endval_block[1:-1]:
            [lhs,rhs] = ig.instruction.split("=")
            end_values[eval(lhs)] = eval(rhs) 
    
    # Now we read the shocks block
    covariances = None
    if shocks_block != None:
        covariances = zeros(len(shocks))
        regex1 = re.compile("var (.*?),(.*?)=(.*)|var (.*?)=(.*)")
        for ig in shocks_block[1:-1]:
            m = regex1.match(ig.instruction)
            if not m:
                raise Exception("unrecognized instruction in block shocks : " + str(ig.instruction))
            if m.group(1) != None:
                varname1 = m.group(1).strip()
                varname2 = m.group(2).strip()
                value = m.group(3).strip().replace("^","**")
            elif m.group(4) != None:
                varname1 = m.group(4).strip()
                varname2 = varname1
                value = m.group(5).strip().replace("^","**")
            i = varexo_names.index(varname1)
            j = varexo_names.index(varname2)
            covariances[i,j] = eval(value,parse_dict)
            covariances[j,i] = eval(value,parse_dict)


    resp = dict()
    resp['variables'] = variables
    resp['parameters'] = parameters
    resp['shocks'] = shocks
    resp['equations'] = equations
    resp['parameters_values'] = parameters_values
    resp['init_values'] = init_values
    resp['covariances'] = covariances

    # Now, let create the objects and inject them into the workspace
#    frame = inspect.currentframe().f_back
#    frame.f_globals['variables'] = variables
#    exovariables = []
#    frame.f_globals['exovariables'] = exovariables
#    frame.f_globals['parameters'] = parameters
#    frame.f_globals['shocks'] = shocks
#    frame.f_globals['equations'] = equations
#    frame.f_globals['parameters_values'] = parameters_values
#    frame.f_globals['init_values'] = init_values
#    frame.f_globals['covariances'] = covariances
    
    # add variables to workspace
#    for v in (variables+shocks+parameters):
#        frame.f_globals[v.name] = v
    
#
    if add_model:
        if model_tags.has_key('fname'):
            fname = model_tags['fname']
        elif not fname:
            fname = "anonymous"
        model = DynareModel(fname,equations,lookup=False)
        model.tags = model_tags
        #model.variables = variables
        #model.shocks = shocks
        #model.parameters = parameters

        model.parameters_ordering = parameters
        model.variables_ordering = variables
        model.shocks_ordering = shocks
        model.equations = equations
        model.parameters_values = parameters_values
        model.init_values = init_values
        model.covariances = covariances
        
        resp['model'] = model
        model.check(verbose = True)

    if full_output == True:
        return resp
    else:
        return model


def dynare_import(filename,names_dict={},full_output=False):
    '''Imports model defined in specified file'''
    import os
    basename = os.path.basename(filename)
    fname = re.compile('(.*)\.mod').match(basename).group(1)
    f = file(filename)
    txt = f.read()
    resp = parse_dynare_text(txt,names_dict=names_dict,full_output=full_output)
    resp.fname = fname
    return resp

    
def undeclare_variables_not_in_equations():
    frame = inspect.currentframe().f_back
    equations = frame.f_globals['equations']
    variables = frame.f_globals['variables']
    init_values = frame.f_globals['init_values']
    l = set()
    for eq in equations:
        l=l.union(  [v.P for v in eq.variables] )
    absent_variables = [v for v in variables if not v in l]
    for v in absent_variables:
        variables.remove(v)
        if v in init_values:
            init_values.pop(v)
    del(frame)