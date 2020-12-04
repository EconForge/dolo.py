def test_grids():

    from dolo.numeric.grids import (
        UniformCartesianGrid,
        UnstructuredGrid,
        NonUniformCartesianGrid,
        SmolyakGrid,
    )
    from dolo.numeric.grids import nodes, n_nodes, node

    print("Cartsian Grid")
    grid = UniformCartesianGrid([0.1, 0.3], [9, 0.4], [50, 10])
    print(grid.nodes)
    print(nodes(grid))

    print("UnstructuredGrid")
    ugrid = UnstructuredGrid([[0.1, 0.3], [9, 0.4], [50, 10]])
    print(nodes(ugrid))
    print(node(ugrid, 0))
    print(n_nodes(ugrid))

    print("Non Uniform CartesianGrid")
    ugrid = NonUniformCartesianGrid([[0.1, 0.3], [9, 0.4], [50, 10]])
    print(nodes(ugrid))
    print(node(ugrid, 0))
    print(n_nodes(ugrid))

    print("Smolyak Grid")
    sg = SmolyakGrid([0.1, 0.2], [1.0, 2.0], 2)
    print(nodes(sg))
    print(node(sg, 1))
    print(n_nodes(sg))
