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
    # for batch in prange(out_shape[0]):  # Batch dimension                                                                                         |
    #     for row in prange(out_shape[1]):  # Output matrix rows                                                                                    |
    #         for col in prange(out_shape[2]):  # Output matrix columns                                                                             |
    #             # Calculate starting positions in A and B                                                                                         |
    #             a_pos = (                                                                                                                         |
    #                 batch * a_batch_stride + row * a_strides[1]                                                                                   |
    #             )  # Position in matrix A                                                                                                         |
    #             b_pos = (                                                                                                                         |
    #                 batch * b_batch_stride + col * b_strides[2]                                                                                   |
    #             )  # Position in matrix B                                                                                                         |
                                                                                                                                                    |
    #             # Do matrix multiplication for this position                                                                                      |
    #             result = 0.0                                                                                                                      |
    #             for k in range(a_shape[2]):  # Inner product dimension                                                                            |
    #                 # A[batch, row, k] * B[batch, k, col]                                                                                         |
    #                 result += a_storage[a_pos] * b_storage[b_pos]                                                                                 |
    #                 # Move to next element in row/column                                                                                          |
    #                 a_pos += a_strides[2]  # Move along row in A                                                                                  |
    #                 b_pos += b_strides[1]  # Move down column in B                                                                                |
                                                                                                                                                    |
    #             # Store result in output                                                                                                          |
    #             out_pos = (                                                                                                                       |
    #                 batch * out_strides[0] + row * out_strides[1] + col * out_strides[2]                                                          |
    #             )                                                                                                                                 |
    #             out[out_pos] = result                                                                                                             |
    total_elements = out_shape[0] * out_shape[1] * out_shape[2]                                                                                     |
    for idx in prange(total_elements):  # Parallel over all output elements-------------------------------------------------------------------------| #16
        # Calculate batch, row, and column indices                                                                                                  |
        batch = idx // (out_shape[1] * out_shape[2])                                                                                                |
        idx_in_batch = idx % (out_shape[1] * out_shape[2])                                                                                          |
        row = idx_in_batch // out_shape[2]                                                                                                          |
        col = idx_in_batch % out_shape[2]                                                                                                           |
                                                                                                                                                    |
        # Calculate starting positions                                                                                                              |
        a_pos = batch * (a_strides[0] if a_shape[0] > 1 else 0) + row * a_strides[1]                                                                |
        b_pos = batch * (b_strides[0] if b_shape[0] > 1 else 0) + col * b_strides[2]                                                                |
                                                                                                                                                    |
        # Compute the dot product                                                                                                                   |
        result = 0.0                                                                                                                                |
        for k in range(a_shape[2]):  # Shared dimension                                                                                             |
            result += (                                                                                                                             |
                a_storage[a_pos + k * a_strides[2]] *                                                                                               |
                b_storage[b_pos + k * b_strides[1]]                                                                                                 |
            )                                                                                                                                       |
                                                                                                                                                    |
        # Store the result                                                                                                                          |
        out_pos = batch * out_strides[0] + row * out_strides[1] + col * out_strides[2]                                                              |
        out[out_pos] = result                                                                                                                       |
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
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #16).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```


## Taining results
### Split Data Set
#### CPU on Google Colab
```bash
python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
```
Total time:  36.01470446586609 Time per epoch:  0.07202940893173218
```
Epoch  0  loss  8.095775919114038 correct 34 time 30.15441131591797
Epoch  10  loss  6.404856188759078 correct 36 time 0.12543463706970215
Epoch  20  loss  4.507042881503628 correct 39 time 0.10784506797790527
Epoch  30  loss  4.665840784964923 correct 42 time 0.10622596740722656
Epoch  40  loss  4.3279284823559445 correct 38 time 0.10654520988464355
Epoch  50  loss  2.259527968008715 correct 47 time 0.10484647750854492
Epoch  60  loss  2.678151867350489 correct 47 time 0.10372352600097656
Epoch  70  loss  2.5692485865267094 correct 48 time 0.11755847930908203
Epoch  80  loss  1.2731688843773559 correct 44 time 0.10500001907348633
Epoch  90  loss  3.61960679481697 correct 44 time 0.10384273529052734
Epoch  100  loss  0.83862061834311 correct 44 time 0.10345458984375
Epoch  110  loss  1.8016460532542122 correct 49 time 0.10584115982055664
Epoch  120  loss  1.3668410498836097 correct 48 time 0.2287883758544922
Epoch  130  loss  2.1143969344094207 correct 46 time 0.10394692420959473
Epoch  140  loss  2.984961813167028 correct 48 time 0.11256122589111328
Epoch  150  loss  4.265965103824842 correct 43 time 0.11532068252563477
Epoch  160  loss  0.7809798299969619 correct 50 time 0.10577774047851562
Epoch  170  loss  1.3613923188942612 correct 50 time 0.10855436325073242
Epoch  180  loss  1.1019922338432877 correct 48 time 0.1154632568359375
Epoch  190  loss  2.4023343775519272 correct 45 time 0.11523008346557617
Epoch  200  loss  1.606807397923785 correct 48 time 0.1255173683166504
Epoch  210  loss  1.3079330075208064 correct 50 time 0.1216893196105957
Epoch  220  loss  1.2986742073145339 correct 50 time 0.12103056907653809
Epoch  230  loss  0.7081603598742303 correct 50 time 0.21445417404174805
Epoch  240  loss  1.121683422902622 correct 50 time 0.1573176383972168
Epoch  250  loss  0.9572405490080286 correct 50 time 0.10739278793334961
Epoch  260  loss  0.8701835791876852 correct 50 time 0.10966968536376953
Epoch  270  loss  0.806390006501512 correct 48 time 0.10597062110900879
Epoch  280  loss  0.14161551311482853 correct 50 time 0.10725522041320801
Epoch  290  loss  0.3434663677357743 correct 49 time 0.1067509651184082
Epoch  300  loss  0.28088479247282144 correct 48 time 0.10685086250305176
Epoch  310  loss  1.7722990604398317 correct 45 time 0.10616898536682129
Epoch  320  loss  1.2062818232643349 correct 48 time 0.10857558250427246
Epoch  330  loss  0.42208316298086007 correct 50 time 0.10545921325683594
Epoch  340  loss  0.6719126152481674 correct 50 time 0.21122241020202637
Epoch  350  loss  2.1048229649889842 correct 45 time 0.1831073760986328
Epoch  360  loss  0.13588030551676789 correct 50 time 0.10961461067199707
Epoch  370  loss  0.20152036970878154 correct 50 time 0.10580897331237793
Epoch  380  loss  0.9297271169968877 correct 50 time 0.10461664199829102
Epoch  390  loss  0.30065906004218285 correct 50 time 0.11615347862243652
Epoch  400  loss  1.2248828967128942 correct 48 time 0.10589885711669922
Epoch  410  loss  0.5293400944061464 correct 50 time 0.10545182228088379
Epoch  420  loss  1.175379054208688 correct 49 time 0.10397171974182129
Epoch  430  loss  0.10521478438417595 correct 50 time 0.11105728149414062
Epoch  440  loss  0.19794942148881967 correct 50 time 0.10425615310668945
Epoch  450  loss  0.5766891378023107 correct 50 time 0.1046745777130127
Epoch  460  loss  0.46727891795711557 correct 49 time 0.18156003952026367
Epoch  470  loss  0.44777877582727754 correct 50 time 0.10518145561218262
Epoch  480  loss  0.554591429498451 correct 50 time 0.10449075698852539
Epoch  490  loss  0.049879573578225596 correct 48 time 0.10316348075866699
```

### GPU on Google Colab
Interestingly it is slightly slower than the CPU version, but still relatively fast
```bash
python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
```
Total time:  78.18348574638367 Time per epoch:  0.15636697149276732
```
Epoch  0  loss  6.893323090591927 correct 31 time 5.915229558944702
Epoch  10  loss  5.153401516572469 correct 40 time 1.3920753002166748
Epoch  20  loss  5.178739942618841 correct 38 time 1.3809609413146973
Epoch  30  loss  3.682146133378537 correct 42 time 1.39158296585083
Epoch  40  loss  3.6398271801449 correct 45 time 1.3970589637756348
Epoch  50  loss  3.2773704490613444 correct 45 time 1.9466986656188965
Epoch  60  loss  5.6021111351253765 correct 40 time 1.3639559745788574
Epoch  70  loss  3.1288228923231967 correct 47 time 1.3829905986785889
Epoch  80  loss  1.7199835540252788 correct 48 time 1.4049761295318604
Epoch  90  loss  2.2512923445857695 correct 49 time 1.3718445301055908
Epoch  100  loss  2.5707567006547904 correct 50 time 1.3510804176330566
Epoch  110  loss  2.3705040314628993 correct 48 time 2.048581123352051
Epoch  120  loss  4.070878967439151 correct 47 time 1.3767390251159668
Epoch  130  loss  2.523684948218698 correct 46 time 1.3432574272155762
Epoch  140  loss  1.872354690146879 correct 47 time 1.3485310077667236
Epoch  150  loss  2.1904683255479074 correct 49 time 1.3800461292266846
Epoch  160  loss  1.7088686835108955 correct 47 time 1.3536596298217773
Epoch  170  loss  2.123208682417246 correct 47 time 2.0172958374023438
Epoch  180  loss  1.2266403226558213 correct 49 time 1.3829891681671143
Epoch  190  loss  0.7715021717239131 correct 47 time 1.358022689819336
Epoch  200  loss  1.418754097069722 correct 49 time 1.3542397022247314
Epoch  210  loss  3.5421018281718983 correct 44 time 1.4087235927581787
Epoch  220  loss  0.6400486295142576 correct 48 time 1.3557860851287842
Epoch  230  loss  1.4024837969680242 correct 49 time 2.011023998260498
Epoch  240  loss  2.7082715238046027 correct 48 time 1.3813073635101318
Epoch  250  loss  1.339123427775952 correct 49 time 1.3877308368682861
Epoch  260  loss  0.42470915136637116 correct 48 time 1.3670918941497803
Epoch  270  loss  0.5521798495062684 correct 47 time 1.360790491104126
Epoch  280  loss  0.9942622560352403 correct 48 time 1.3670272827148438
Epoch  290  loss  0.6921717363197064 correct 46 time 2.045898199081421
Epoch  300  loss  0.2921331841370231 correct 48 time 1.3562450408935547
Epoch  310  loss  2.1750131100840537 correct 46 time 1.3559463024139404
Epoch  320  loss  1.0985812504884445 correct 47 time 1.3863635063171387
Epoch  330  loss  0.44688337717393756 correct 48 time 1.3494055271148682
Epoch  340  loss  1.431394135505046 correct 49 time 1.3593909740447998
Epoch  350  loss  1.0369026254883849 correct 50 time 2.0049071311950684
Epoch  360  loss  0.08837916792877056 correct 49 time 1.3521075248718262
Epoch  370  loss  0.6585963185824999 correct 48 time 1.3565161228179932
Epoch  380  loss  0.4305331058333165 correct 49 time 1.3815944194793701
Epoch  390  loss  0.6436017344323391 correct 48 time 1.3327093124389648
Epoch  400  loss  0.7347775902964946 correct 49 time 1.3453290462493896
Epoch  410  loss  0.5554595395013072 correct 49 time 2.029207229614258
Epoch  420  loss  1.8439699456238996 correct 49 time 1.3508367538452148
Epoch  430  loss  2.2177538514991273 correct 47 time 1.3776836395263672
Epoch  440  loss  0.1575825470390596 correct 49 time 1.3800580501556396
Epoch  450  loss  0.16283225291676312 correct 50 time 1.359158992767334
Epoch  460  loss  2.3813081196626538 correct 47 time 1.3654754161834717
Epoch  470  loss  1.2770170760116515 correct 50 time 2.0853188037872314
Epoch  480  loss  1.4210888445448813 correct 48 time 1.3508849143981934
Epoch  490  loss  0.0911259922986424 correct 48 time 1.3571515083312988
```

### XOR
#### CPU
```bash
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET xor --RATE 0.05
```
Total time:  43.853522300720215 Time per epoch:  0.08770704460144042
```
Epoch  0  loss  5.574076201601594 correct 32 time 31.409825801849365
Epoch  10  loss  3.074798879528437 correct 43 time 0.2168893814086914
Epoch  20  loss  5.820065999787587 correct 47 time 0.220109224319458
Epoch  30  loss  2.6744211944640015 correct 45 time 0.22270989418029785
Epoch  40  loss  0.8090898279208787 correct 47 time 0.23153471946716309
Epoch  50  loss  0.962328751389861 correct 49 time 0.43915867805480957
Epoch  60  loss  0.9415192349610831 correct 48 time 0.22221088409423828
Epoch  70  loss  1.0173522933664925 correct 45 time 0.22536277770996094
Epoch  80  loss  1.3286311747212731 correct 49 time 0.22065997123718262
Epoch  90  loss  1.4579769954591175 correct 49 time 0.21808195114135742
Epoch  100  loss  1.7620109505153119 correct 50 time 0.21685361862182617
Epoch  110  loss  1.7322186178710917 correct 49 time 0.21599507331848145
Epoch  120  loss  1.5483955047514748 correct 50 time 0.22984790802001953
Epoch  130  loss  1.198528544462571 correct 50 time 0.2177293300628662
Epoch  140  loss  0.8248581561662948 correct 49 time 0.23309016227722168
Epoch  150  loss  0.5403504966289387 correct 50 time 0.23737645149230957
Epoch  160  loss  1.111807057733327 correct 50 time 0.5321197509765625
Epoch  170  loss  0.5143410511019326 correct 49 time 0.22326993942260742
Epoch  180  loss  0.8281146431441961 correct 50 time 0.22282099723815918
Epoch  190  loss  0.5915111216481241 correct 50 time 0.22746753692626953
Epoch  200  loss  0.9052295901826902 correct 50 time 0.2205371856689453
Epoch  210  loss  0.07178335791037624 correct 50 time 0.4425535202026367
Epoch  220  loss  0.20518572646456712 correct 50 time 0.21832942962646484
Epoch  230  loss  0.9908858834277205 correct 50 time 0.21840882301330566
Epoch  240  loss  0.550582290225132 correct 50 time 0.235975980758667
Epoch  250  loss  0.46384003612254043 correct 50 time 0.23838305473327637
Epoch  260  loss  0.28310821521032453 correct 50 time 0.23926067352294922
Epoch  270  loss  0.34734727510035046 correct 50 time 0.21737217903137207
Epoch  280  loss  0.5755146953462991 correct 50 time 0.21663784980773926
Epoch  290  loss  0.22235078661472277 correct 50 time 0.22089290618896484
Epoch  300  loss  0.3030587836623519 correct 50 time 0.21747922897338867
Epoch  310  loss  0.5630699359488683 correct 50 time 0.21710586547851562
Epoch  320  loss  0.25564485902683926 correct 50 time 0.42017674446105957
Epoch  330  loss  0.25410662727937494 correct 50 time 0.22146105766296387
Epoch  340  loss  0.04872567017908132 correct 50 time 0.2182450294494629
Epoch  350  loss  0.2473648963214245 correct 50 time 0.24322271347045898
Epoch  360  loss  0.4931933691477947 correct 50 time 0.22597551345825195
Epoch  370  loss  0.5327159917706091 correct 50 time 0.5220317840576172
Epoch  380  loss  0.2667538067444244 correct 50 time 0.22245287895202637
Epoch  390  loss  0.1398749062435736 correct 50 time 0.22922945022583008
Epoch  400  loss  0.30648020741045123 correct 50 time 0.2200319766998291
Epoch  410  loss  0.16023422687580213 correct 50 time 0.21880078315734863
Epoch  420  loss  0.3379076755481595 correct 50 time 0.4238719940185547
Epoch  430  loss  0.18965759237366167 correct 50 time 0.23363065719604492
Epoch  440  loss  0.3131568406059659 correct 50 time 0.23661375045776367
Epoch  450  loss  0.1297821858136073 correct 50 time 0.24257898330688477
Epoch  460  loss  0.038079933520805465 correct 50 time 0.23586797714233398
Epoch  470  loss  0.41266283740977955 correct 50 time 0.21861577033996582
Epoch  480  loss  0.43425647650099336 correct 50 time 0.2155933380126953
Epoch  490  loss  0.24630950481846442 correct 50 time 0.2190711498260498
```

#### GPU
```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 200 --DATASET xor --RATE 0.05 --PLOT true
```

Total time:  80.090252161026 Time per epoch:  0.160180504322052
```
Epoch  0  loss  10.367326224800458 correct 29 time 4.722745418548584
Epoch  10  loss  9.75769328057765 correct 35 time 1.634993553161621
Epoch  20  loss  3.9695695074356383 correct 42 time 1.485105276107788
Epoch  30  loss  3.5325496107339136 correct 44 time 1.4007010459899902
Epoch  40  loss  5.7239824229828145 correct 43 time 1.429976224899292
Epoch  50  loss  2.803959675181489 correct 44 time 1.4134495258331299
Epoch  60  loss  4.368098644522748 correct 44 time 2.124481439590454
Epoch  70  loss  2.1389303707969174 correct 45 time 1.4172725677490234
Epoch  80  loss  1.7645825576706935 correct 44 time 1.4045383930206299
Epoch  90  loss  5.158387155313838 correct 46 time 1.4136111736297607
Epoch  100  loss  2.8294054000537603 correct 46 time 1.422145128250122
Epoch  110  loss  1.32920923078834 correct 47 time 1.9664688110351562
Epoch  120  loss  2.565687134298164 correct 44 time 1.4144670963287354
Epoch  130  loss  2.690328088743282 correct 48 time 1.4484920501708984
Epoch  140  loss  1.8641236860028187 correct 49 time 1.430870532989502
Epoch  150  loss  3.0168005567882723 correct 50 time 1.6757938861846924
Epoch  160  loss  2.577984587731726 correct 45 time 1.5029051303863525
Epoch  170  loss  1.4329753490412984 correct 50 time 1.4310407638549805
Epoch  180  loss  2.360679428053495 correct 48 time 1.4008052349090576
Epoch  190  loss  1.989996361625387 correct 50 time 1.4256722927093506
Epoch  200  loss  1.198531924672648 correct 49 time 1.9166457653045654
Epoch  210  loss  0.9015590160404114 correct 50 time 1.4605047702789307
Epoch  220  loss  1.415411447956744 correct 50 time 1.4164228439331055
Epoch  230  loss  1.1063752790253938 correct 50 time 1.446260690689087
Epoch  240  loss  0.2841987054846418 correct 48 time 1.4083364009857178
Epoch  250  loss  0.9037077015490945 correct 49 time 2.175642728805542
Epoch  260  loss  1.5251406543637394 correct 45 time 1.4106554985046387
Epoch  270  loss  2.466163743514401 correct 47 time 1.407989740371704
Epoch  280  loss  1.6646061792096445 correct 50 time 1.4333033561706543
Epoch  290  loss  1.6917928663222517 correct 50 time 1.3952524662017822
Epoch  300  loss  1.5778005347827941 correct 50 time 2.0164408683776855
Epoch  310  loss  0.5949838132608596 correct 50 time 1.4218502044677734
Epoch  320  loss  1.2294661719049984 correct 50 time 1.4182283878326416
Epoch  330  loss  1.1751821269735423 correct 48 time 1.4245929718017578
Epoch  340  loss  1.7954564441060854 correct 47 time 1.4505500793457031
Epoch  350  loss  0.20152222330927344 correct 50 time 1.874955177307129
Epoch  360  loss  0.8236280146835429 correct 50 time 1.413541316986084
Epoch  370  loss  0.4971501495116969 correct 50 time 1.4245855808258057
Epoch  380  loss  0.8375654356377014 correct 50 time 1.4248864650726318
Epoch  390  loss  1.0030108546933876 correct 50 time 1.5917015075683594
Epoch  400  loss  0.7690074145689618 correct 50 time 1.5902745723724365
Epoch  410  loss  1.0475284384114525 correct 50 time 1.433666467666626
Epoch  420  loss  0.26637815281227895 correct 50 time 1.4085586071014404
Epoch  430  loss  0.892497684103851 correct 50 time 1.408045768737793
Epoch  440  loss  0.7530691449740256 correct 50 time 1.784693956375122
Epoch  450  loss  0.3484495822476382 correct 50 time 1.4094552993774414
Epoch  460  loss  0.32811024857273435 correct 50 time 1.4631962776184082
Epoch  470  loss  0.14309951134259624 correct 50 time 1.4647133350372314
Epoch  480  loss  0.47373589712694925 correct 50 time 1.4504315853118896
Epoch  490  loss  0.4337489274331194 correct 50 time 2.1793339252471924
````