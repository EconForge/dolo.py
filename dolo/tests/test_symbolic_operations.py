def test_shift_time():

    from dolo.compiler.function_compiler_ast import timeshift, to_source, to_expr
    import ast

    s = 'a + b + c(1)'
    
    assert( timeshift(s, ['c'], -1) == 'a + b + c')
    assert( timeshift(s, ['a', 'c'], -1) == 'a(-(1)) + b + c')
    assert( timeshift(s, ['b'], 1) == 'a + b(1) + c(1)')

def test_steady_state():

    from dolo.compiler.function_compiler_ast import timeshift, to_source, to_expr
    import ast

    s = 'a + b + c(1)'

    assert( timeshift(s, ['c'], 'S') == 'a + b + c')
    assert( timeshift(s, ['a', 'c'], 'S') == 'a + b + c')
    assert( timeshift(s, ['b'], 'S') == 'a + b + c(1)')


if __name__ == '__main__':

    test_shift_time()
    test_steady_state()
