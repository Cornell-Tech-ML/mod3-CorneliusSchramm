# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

## Parallelization Diagnostics Output:

```
(.venv) ➜  mod3-CorneliusSchramm git:(master) python project/parallel_check.py

MAP
OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/corneliusschramm/Documents/Github/Cornell Tech/Semester
1/MLE/mod3-CorneliusSchramm/minitorch/fast_ops.py (167)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/corneliusschramm/Documents/Github/Cornell Tech/Semester 1/MLE/mod3-CorneliusSchramm/minitorch/fast_ops.py (167)
-----------------------------------------------------------------------------------|loop #ID
    def _map(                                                                      |
        out: Storage,                                                              |
        out_shape: Shape,                                                          |
        out_strides: Strides,                                                      |
        in_storage: Storage,                                                       |
        in_shape: Shape,                                                           |
        in_strides: Strides,                                                       |
    ) -> None:                                                                     |
        # TODO: Implement for Task 3.1.                                            |
        # ASSIGN2.2:                                                               |
        if (                                                                       |
            len(out_strides) != len(in_strides)                                    |
            or (out_strides != in_strides).any()-----------------------------------| #0
            or (out_shape != in_shape).any()---------------------------------------| #1
        ):                                                                         |
            for i in prange(len(out)):---------------------------------------------| #5
                out_index: Index = np.zeros(MAX_DIMS, np.int16)  # type: ignore----| #2
                in_index: Index = np.zeros(MAX_DIMS, np.int16)  # type: ignore-----| #3
                to_index(i, out_shape, out_index)                                  |
                broadcast_index(out_index, out_shape, in_shape, in_index)          |
                o = index_to_position(out_index, out_strides)                      |
                j = index_to_position(in_index, in_strides)                        |
                out[o] = fn(in_storage[j])                                         |
        else:                                                                      |
            # When `out` and `in` are stride-aligned, avoid indexing               |
            for i in prange(len(out)):---------------------------------------------| #4
                out[i] = fn(in_storage[i])                                         |
        # END ASSIGN2.2                                                            |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...

Fused loop summary:
+--2 has the following loops fused into it:
   +--3 (fused)
Following the attempted fusion of parallel for-loops there are 5 parallel for-
loop(s) (originating from loops labelled: #0, #1, #5, #2, #4).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--5 is a parallel loop
   +--2 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--5 (parallel)
   +--2 (parallel)
   +--3 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--5 (parallel)
   +--2 (serial, fused with loop(s): 3)



Parallel region 0 (loop #5) had 1 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#5).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/corneliusschramm/Documents/Github/Cornell Tech/Semester
1/MLE/mod3-CorneliusSchramm/minitorch/fast_ops.py (183) is hoisted out of the
parallel loop labelled #5 (it will be performed before the loop is executed and
reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, np.int16)  # type: ignore
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/corneliusschramm/Documents/Github/Cornell Tech/Semester
1/MLE/mod3-CorneliusSchramm/minitorch/fast_ops.py (184) is hoisted out of the
parallel loop labelled #5 (it will be performed before the loop is executed and
reused inside the loop):
   Allocation:: in_index: Index = np.zeros(MAX_DIMS, np.int16)  # type: ignore
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/corneliusschramm/Documents/Github/Cornell Tech/Semester
1/MLE/mod3-CorneliusSchramm/minitorch/fast_ops.py (222)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/corneliusschramm/Documents/Github/Cornell Tech/Semester 1/MLE/mod3-CorneliusSchramm/minitorch/fast_ops.py (222)
---------------------------------------------------------------------------|loop #ID
    def _zip(                                                              |
        out: Storage,                                                      |
        out_shape: Shape,                                                  |
        out_strides: Strides,                                              |
        a_storage: Storage,                                                |
        a_shape: Shape,                                                    |
        a_strides: Strides,                                                |
        b_storage: Storage,                                                |
        b_shape: Shape,                                                    |
        b_strides: Strides,                                                |
    ) -> None:                                                             |
        if (                                                               |
            len(out_strides) != len(a_strides)                             |
            or len(out_strides) != len(b_strides)                          |
            or (out_strides != a_strides).any()----------------------------| #6
            or (out_strides != b_strides).any()----------------------------| #7
            or (out_shape != a_shape).any()--------------------------------| #8
            or (out_shape != b_shape).any()--------------------------------| #9
        ):                                                                 |
            for i in prange(len(out)):-------------------------------------| #14
                out_index: Index = np.zeros(MAX_DIMS, np.int32)------------| #10
                a_index: Index = np.zeros(MAX_DIMS, np.int32)--------------| #11
                b_index: Index = np.zeros(MAX_DIMS, np.int32)--------------| #12
                to_index(i, out_shape, out_index)                          |
                o = index_to_position(out_index, out_strides)              |
                broadcast_index(out_index, out_shape, a_shape, a_index)    |
                j = index_to_position(a_index, a_strides)                  |
                broadcast_index(out_index, out_shape, b_shape, b_index)    |
                k = index_to_position(b_index, b_strides)                  |
                out[o] = fn(a_storage[j], b_storage[k])                    |
        else:                                                              |
            # When out, a, b are stride-aligned, avoid indexing            |
            for i in prange(len(out)):-------------------------------------| #13
                out[i] = fn(a_storage[i], b_storage[i])                    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...

Fused loop summary:
+--10 has the following loops fused into it:
   +--11 (fused)
   +--12 (fused)
Following the attempted fusion of parallel for-loops there are 7 parallel for-
loop(s) (originating from loops labelled: #6, #7, #8, #9, #14, #10, #13).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--14 is a parallel loop
   +--10 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--14 (parallel)
   +--10 (parallel)
   +--11 (parallel)
   +--12 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--14 (parallel)
   +--10 (serial, fused with loop(s): 11, 12)



Parallel region 0 (loop #14) had 2 loop(s) fused and 1 loop(s) serialized as
part of the larger parallel loop (#14).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/corneliusschramm/Documents/Github/Cornell Tech/Semester
1/MLE/mod3-CorneliusSchramm/minitorch/fast_ops.py (242) is hoisted out of the
parallel loop labelled #14 (it will be performed before the loop is executed and
 reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/corneliusschramm/Documents/Github/Cornell Tech/Semester
1/MLE/mod3-CorneliusSchramm/minitorch/fast_ops.py (243) is hoisted out of the
parallel loop labelled #14 (it will be performed before the loop is executed and
 reused inside the loop):
   Allocation:: a_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/corneliusschramm/Documents/Github/Cornell Tech/Semester
1/MLE/mod3-CorneliusSchramm/minitorch/fast_ops.py (244) is hoisted out of the
parallel loop labelled #14 (it will be performed before the loop is executed and
 reused inside the loop):
   Allocation:: b_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/corneliusschramm/Documents/Github/Cornell Tech/Semester
1/MLE/mod3-CorneliusSchramm/minitorch/fast_ops.py (281)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/corneliusschramm/Documents/Github/Cornell Tech/Semester 1/MLE/mod3-CorneliusSchramm/minitorch/fast_ops.py (281)
-------------------------------------------------------------|loop #ID
    def _reduce(                                             |
        out: Storage,                                        |
        out_shape: Shape,                                    |
        out_strides: Strides,                                |
        a_storage: Storage,                                  |
        a_shape: Shape,                                      |
        a_strides: Strides,                                  |
        reduce_dim: int,                                     |
    ) -> None:                                               |
        for i in prange(len(out)):---------------------------| #15
            out_index = np.empty(MAX_DIMS, np.int32)         |
            size = a_shape[reduce_dim]  # the reduce size    |
            # get the index of i                             |
            to_index(i, out_shape, out_index)                |
            # get the position                               |
            o = index_to_position(out_index, out_strides)    |
            j = index_to_position(out_index, a_strides)      |
            # the accumulation                               |
            a = out[o]                                       |
            step = a_strides[reduce_dim]                     |
            for s in range(size):                            |
                a = fn(a, a_storage[j])                      |
                j += step                                    |
            out[o] = a                                       |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #15).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/corneliusschramm/Documents/Github/Cornell Tech/Semester
1/MLE/mod3-CorneliusSchramm/minitorch/fast_ops.py (291) is hoisted out of the
parallel loop labelled #15 (it will be performed before the loop is executed and
 reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/corneliusschramm/Documents/Github/Cornell Tech/Semester
1/MLE/mod3-CorneliusSchramm/minitorch/fast_ops.py (309)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/corneliusschramm/Documents/Github/Cornell Tech/Semester 1/MLE/mod3-CorneliusSchramm/minitorch/fast_ops.py (309)
----------------------------------------------------------------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                                                                        |
    out: Storage,                                                                                                                                   |
    out_shape: Shape,                                                                                                                               |
    out_strides: Strides,                                                                                                                           |
    a_storage: Storage,                                                                                                                             |
    a_shape: Shape,                                                                                                                                 |
    a_strides: Strides,                                                                                                                             |
    b_storage: Storage,                                                                                                                             |
    b_shape: Shape,                                                                                                                                 |
    b_strides: Strides,                                                                                                                             |
) -> None:                                                                                                                                          |
    """NUMBA tensor matrix multiply function.                                                                                                       |
                                                                                                                                                    |
    Should work for any tensor shapes that broadcast as long as                                                                                     |
                                                                                                                                                    |
    ```                                                                                                                                             |
    assert a_shape[-1] == b_shape[-2]                                                                                                               |
    ```                                                                                                                                             |
                                                                                                                                                    |
    Optimizations:                                                                                                                                  |
                                                                                                                                                    |
    * Outer loop in parallel                                                                                                                        |
    * No index buffers or function calls                                                                                                            |
    * Inner loop should have no global writes, 1 multiply.                                                                                          |
                                                                                                                                                    |
                                                                                                                                                    |
    Args:                                                                                                                                           |
    ----                                                                                                                                            |
        out (Storage): storage for `out` tensor                                                                                                     |
        out_shape (Shape): shape for `out` tensor                                                                                                   |
        out_strides (Strides): strides for `out` tensor                                                                                             |
        a_storage (Storage): storage for `a` tensor                                                                                                 |
        a_shape (Shape): shape for `a` tensor                                                                                                       |
        a_strides (Strides): strides for `a` tensor                                                                                                 |
        b_storage (Storage): storage for `b` tensor                                                                                                 |
        b_shape (Shape): shape for `b` tensor                                                                                                       |
        b_strides (Strides): strides for `b` tensor                                                                                                 |
                                                                                                                                                    |
    Returns:                                                                                                                                        |
    -------                                                                                                                                         |
        None : Fills in `out`                                                                                                                       |
                                                                                                                                                    |
    """                                                                                                                                             |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                                                                          |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                                                                          |
    # Parallel loop through batches and matrix dimensions                                                                                           |
    for batch in prange(out_shape[0]):  # Batch dimension-------------------------------------------------------------------------------------------| #18
        for row in prange(out_shape[1]):  # Output matrix rows--------------------------------------------------------------------------------------| #17
            for col in prange(out_shape[2]):  # Output matrix columns-------------------------------------------------------------------------------| #16
                # Calculate starting positions in A and B                                                                                           |
                a_pos = (                                                                                                                           |
                    batch * a_batch_stride + row * a_strides[1]                                                                                     |
                )  # Position in matrix A                                                                                                           |
                b_pos = (                                                                                                                           |
                    batch * b_batch_stride + col * b_strides[2]                                                                                     |
                )  # Position in matrix B                                                                                                           |
                                                                                                                                                    |
                # Do matrix multiplication for this position                                                                                        |
                result = 0.0                                                                                                                        |
                for k in range(a_shape[2]):  # Inner product dimension                                                                              |
                    # A[batch, row, k] * B[batch, k, col]                                                                                           |
                    result += a_storage[a_pos] * b_storage[b_pos]                                                                                   |
                    # Move to next element in row/column                                                                                            |
                    a_pos += a_strides[2]  # Move along row in A                                                                                    |
                    b_pos += b_strides[1]  # Move down column in B                                                                                  |
                                                                                                                                                    |
                # Store result in output                                                                                                            |
                out_pos = (                                                                                                                         |
                    batch * out_strides[0] + row * out_strides[1] + col * out_strides[2]                                                            |
                )                                                                                                                                   |
                out[out_pos] = result                                                                                                               |
                                                                                                                                                    |
    # A:                                                                                                                                            |
    # [                                                                                                                                             |
    #     [1 ,2 ]                                                                                                                                   |
    #     [3, 4]                                                                                                                                    |
    # ]                                                                                                                                             |
    # B:                                                                                                                                            |
    # [                                                                                                                                             |
    #     [5, 6]                                                                                                                                    |
    #     [7, 8]                                                                                                                                    |
    # ]                                                                                                                                             |
    # A @ B:                                                                                                                                        |
    # [                                                                                                                                             |
    #     [1*5 + 2*7, 1*6 + 2*8]                                                                                                                    |
    #     [3*5 + 4*7, 3*6 + 4*8]                                                                                                                    |
    # ]                                                                                                                                             |
    # spelled out in words                                                                                                                          |
    # A @ B:                                                                                                                                        |
    # [                                                                                                                                             |
    #     [ A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0], A[0, 0] * B[0, 1] + A[0, 1] * B[1, 1] ]                                                          |
    #     [ A[1, 0] * B[0, 0] + A[1, 1] * B[1, 0], A[1, 0] * B[0, 1] + A[1, 1] * B[1, 1] ]                                                          |
    # ]                                                                                                                                             |
    # on a basic level with for loops                                                                                                               |
    # for row_a in range(A.rows):                                                                                                                   |
    #     for col_b in range(B.cols):                                                                                                               |
    #         accumulator = 0                                                                                                                       |
    #         for col_a in range(A.cols):                                                                                                           |
    #              accumulator += A[row_a, col_a] * B[col_a, col_b]                                                                                 |
    #         out[row_a, col_b] = accumulator                                                                                                       |
    # since we start looping through out storage, we need to basically work backwards from storage position to the index out[i, j] or something?    |
    # for out_ordinal in prange(len(out)):                                                                                                          |
    #     # find index in out                                                                                                                       |
    #     out_index = np.zeros(2, np.int32)                                                                                                         |
    #     to_index(out_ordinal, out_shape, out_index)                                                                                               |
                                                                                                                                                    |
    #     #  out index                                                                                                                              |
    #     row_a, col_b = out_index[0], out_index[1]                                                                                                 |
                                                                                                                                                    |
    #     # accumulator                                                                                                                             |
    #     accumulator = 0                                                                                                                           |
    #     a_index = np.array([row_a, 0], np.int32)                                                                                                  |
    #     b_index = np.array([0, col_b], np.int32)                                                                                                  |
    #     for inner in range(a_shape[-1]):                                                                                                          |
    #         a_index[1] = inner                                                                                                                    |
    #         b_index[0] = inner                                                                                                                    |
    #         a_ordinal = index_to_position(a_index, a_strides)                                                                                     |
    #         b_ordinal = index_to_position(b_index, b_strides)                                                                                     |
    #         accumulator += a_storage[a_ordinal] * b_storage[b_ordinal]                                                                            |
    #     out[out_ordinal] = accumulator                                                                                                            |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #18, #17).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--18 is a parallel loop
   +--17 --> rewritten as a serial loop
      +--16 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--18 (parallel)
   +--17 (parallel)
      +--16 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--18 (parallel)
   +--17 (serial)
      +--16 (serial)



Parallel region 0 (loop #18) had 0 loop(s) fused and 2 loop(s) serialized as
part of the larger parallel loop (#18).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```
Note, I found the diagnostics output a bit hard to interpret which is why I asked o1 to do it for me. It seems like thins
MAP and ZIP Functions
For both the MAP and ZIP functions, Numba provides detailed optimization reports:

Loop Fusion: Numba successfully fused and optimized loops to improve performance.
Allocation Hoisting: It hoisted memory allocations out of the parallel loops, reducing overhead.
Parallelization: The loops are effectively parallelized, as indicated by the parallel loop IDs (#5 for MAP and #14 for ZIP).
REDUCE Function
For the REDUCE function:

Limited Optimization: Numba recognizes the parallel loop (#15) but doesn't apply loop fusion or other optimizations.
Reason: Reduction operations involve accumulating results, which can be challenging to parallelize due to dependencies between iterations.
Allocation Hoisting: Numba hoisted the allocation of out_index out of the loop, which is beneficial.
MATRIX MULTIPLY Function
In the MATRIX MULTIPLY section:

Parallelization Strategy: Numba parallelizes the outermost loop (#18), which iterates over batches.
Inner Loops: The inner loops (#17 and #16) are serialized to optimize performance and avoid overhead from excessive parallelism.
No Loop Fusion: Numba didn't fuse loops here, likely because matrix multiplication requires careful handling of loop bounds and data dependencies.
Conclusion
Your script appears to be correctly set up:

Numba Diagnostics: The output shows that Numba is analyzing and optimizing your functions appropriately.
Parallelization: Where possible, loops are parallelized and optimized.
Understanding Reduce: The output for the REDUCE function aligns with the inherent challenges of parallelizing reduction operations.