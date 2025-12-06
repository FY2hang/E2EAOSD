def _einsum_flops_compute(equation, *operands):
    """
    Count flops for the einsum operation.
    """
    equation = equation.replace(" ", "")
    input_shapes = []
    for o in operands:
        if type(o) is list:
            for o_ in o:
                input_shapes.append(o_.shape)
        else:
            input_shapes.append(o.shape)

    # Re-map equation so that same equation with different alphabet
    # representations will look the same.
    letter_order = OrderedDict((k, 0) for k in equation if k.isalpha()).keys()
    mapping = {ord(x): 97 + i for i, x in enumerate(letter_order)}
    equation = equation.translate(mapping)

    np_arrs = [np.zeros(s) for s in input_shapes]
    optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
    for line in optim.split("\n"):
        if "optimized flop" in line.lower():
            flop = int(float(line.split(":")[-1]))
            return flop, 0
    raise NotImplementedError("Unsupported einsum operation.")