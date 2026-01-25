from src.matrix import Matrix




def elim(
        matrix: Matrix, group_operations: bool = False, verbose: bool = True, get_inv: bool = False
        ) -> Matrix:
    """Gauss-Jordan elimination, and output result + steps
    Optional Args:
    - group_operations: If True, group row operations together in output
    - verbose: If True, print matrix after (optionally grouped) row operations. 
                Otherwise, only print initial and final matrix.
    - get_inv: If true, print the RREF operations & inverse operations
    Returns the matrix in RREF form.
    """
    if get_inv and matrix.rows != matrix.cols:
        raise ValueError("Matrix must be square to compute inverse.")
    print("Initial Matrix:")
    print(matrix)

    identity_matrix = Matrix.identity(matrix.rows, matrix.cols, matrix.field)
    if get_inv:
        print("Initial Identity Matrix:")
        print(identity_matrix)
    pivot = [0, 0]
    while pivot[0] < matrix.rows and pivot[1] < matrix.cols:
        if matrix.first_nonzero(pivot[1], first_row=pivot[0]) == -1:
            pivot[1] += 1
        else:
            i = matrix.first_nonzero(pivot[1], first_row=pivot[0])
            if i != pivot[0]:
                matrix.swap_rows(i, pivot[0])
                print(f"Swapped Rows: R{i+1} <-> R{pivot[0]+1}")
                if verbose:
                    print(f"Result of ERO:\n{matrix}\n")
                    if get_inv:
                        identity_matrix.swap_rows(i, pivot[0])
                        print(f"Result of ERO on identity matrix:\n{identity_matrix}\n")
            pivot_inverse = matrix[pivot[0], pivot[1]].mult_inv()
            matrix.multiply_row(pivot[0], pivot_inverse)
            print(f"Scaled Row: R{pivot[0]+1} * {pivot_inverse}")
            if get_inv:
                identity_matrix.multiply_row(pivot[0], pivot_inverse)
                print(f"Scaled Row (Identity Matrix): R{pivot[0]+1} * {pivot_inverse}") 
            if verbose:
                print(f"Result of ERO:\n{matrix}\n")
                if get_inv:
                    print(f"Result of ERO on identity matrix:\n{identity_matrix}\n")
            for r in range(matrix.rows):
                if r != pivot[0]:
                    inv = -matrix[r, pivot[1]]
                    matrix.add_multiple_of_row(r, pivot[0], inv)
                    print(f"Added Multiple of Row: R{r+1} + ({inv}) * R{pivot[0]+1}")
                    if get_inv:
                        identity_matrix.add_multiple_of_row(r, pivot[0], inv)
                    if not group_operations and verbose:
                        print(f"Result of ERO:\n{matrix}\n")
                        if get_inv:
                            print(f"Result of ERO on identity matrix:\n{identity_matrix}\n")
            if group_operations and verbose:
                print(f"Result of grouped EROs:\n{matrix}\n")
                if get_inv:
                    print(f"Result of grouped EROs on identity matrix:\n{identity_matrix}\n")
            pivot[0] += 1
            pivot[1] += 1
    print("Final RREF:")
    print(matrix)
    if get_inv:
        print("Final Inverse Matrix:")
        print(identity_matrix)
    return matrix
