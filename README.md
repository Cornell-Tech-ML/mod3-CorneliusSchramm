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
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 150 --DATASET split --RATE 0.05 --PLOT True
```
Total time:  105.2787 Time per epoch:  0.2106
```
Epoch  0  loss  5.9560492357185035 correct 26 total time 23.6339 time per epoch 23.6339
Epoch  10  loss  5.11405367924167 correct 41 total time 25.1207 time per epoch 2.2837
Epoch  20  loss  3.346455625846713 correct 32 total time 26.6262 time per epoch 1.2679
Epoch  30  loss  5.147936838949785 correct 48 total time 28.1264 time per epoch 0.9073
Epoch  40  loss  3.999483016765956 correct 48 total time 29.6731 time per epoch 0.7237
Epoch  50  loss  2.9419908286753764 correct 48 total time 31.169 time per epoch 0.6112
Epoch  60  loss  2.24918359757454 correct 48 total time 32.6744 time per epoch 0.5356
Epoch  70  loss  3.763575533815807 correct 41 total time 34.6982 time per epoch 0.4887
Epoch  80  loss  1.2778150930209504 correct 48 total time 36.7338 time per epoch 0.4535
Epoch  90  loss  0.50638495503805 correct 49 total time 38.2275 time per epoch 0.4201
Epoch  100  loss  1.3990790927951315 correct 48 total time 39.7275 time per epoch 0.3933
Epoch  110  loss  0.9765208303721662 correct 47 total time 41.2333 time per epoch 0.3715
Epoch  120  loss  2.313749290595167 correct 50 total time 42.751 time per epoch 0.3533
Epoch  130  loss  0.9628458936496004 correct 48 total time 44.2412 time per epoch 0.3377
Epoch  140  loss  2.2343380284175547 correct 48 total time 45.7746 time per epoch 0.3246
Epoch  150  loss  1.825545866787625 correct 48 total time 48.4019 time per epoch 0.3205
Epoch  160  loss  0.3474070448657494 correct 48 total time 49.9018 time per epoch 0.3099
Epoch  170  loss  0.7148654142868914 correct 48 total time 51.3868 time per epoch 0.3005
Epoch  180  loss  1.4647745158586203 correct 50 total time 52.8755 time per epoch 0.2921
Epoch  190  loss  0.3797781225965503 correct 49 total time 54.3631 time per epoch 0.2846
Epoch  200  loss  0.8558573055436715 correct 48 total time 55.8697 time per epoch 0.278
Epoch  210  loss  0.2706531199044801 correct 48 total time 57.3639 time per epoch 0.2719
Epoch  220  loss  1.3754693775875617 correct 50 total time 59.6692 time per epoch 0.27
Epoch  230  loss  0.5277738761960337 correct 48 total time 61.5275 time per epoch 0.2664
Epoch  240  loss  1.6743109217801297 correct 48 total time 63.0126 time per epoch 0.2615
Epoch  250  loss  0.18175405512683826 correct 49 total time 64.4982 time per epoch 0.257
Epoch  260  loss  2.6857642309334517 correct 47 total time 65.9911 time per epoch 0.2528
Epoch  270  loss  2.1739467753036577 correct 47 total time 67.4873 time per epoch 0.249
Epoch  280  loss  1.2561321027285033 correct 50 total time 68.9882 time per epoch 0.2455
Epoch  290  loss  1.910357739869383 correct 49 total time 70.6694 time per epoch 0.2429
Epoch  300  loss  0.6989615607902466 correct 48 total time 73.1116 time per epoch 0.2429
Epoch  310  loss  1.6280781020526027 correct 49 total time 74.6143 time per epoch 0.2399
Epoch  320  loss  1.0574105340000897 correct 49 total time 76.2389 time per epoch 0.2375
Epoch  330  loss  1.4824128780613048 correct 48 total time 77.7411 time per epoch 0.2349
Epoch  340  loss  1.1716816168066013 correct 48 total time 79.2538 time per epoch 0.2324
Epoch  350  loss  1.5428271646129563 correct 49 total time 80.7413 time per epoch 0.23
Epoch  360  loss  0.4199198912046277 correct 49 total time 82.2193 time per epoch 0.2278
Epoch  370  loss  1.388550906591196 correct 50 total time 84.6077 time per epoch 0.2281
Epoch  380  loss  0.15481383231053208 correct 49 total time 86.3212 time per epoch 0.2266
Epoch  390  loss  0.38485692586780557 correct 50 total time 87.8114 time per epoch 0.2246
Epoch  400  loss  0.26040791162675736 correct 48 total time 89.3486 time per epoch 0.2228
Epoch  410  loss  1.5164801864213486 correct 49 total time 90.8464 time per epoch 0.221
Epoch  420  loss  0.12178329463144831 correct 50 total time 92.3415 time per epoch 0.2193
Epoch  430  loss  1.226819499527664 correct 49 total time 93.8485 time per epoch 0.2177
Epoch  440  loss  0.11107203969687769 correct 48 total time 95.5991 time per epoch 0.2168
Epoch  450  loss  0.2865555854075013 correct 48 total time 97.9089 time per epoch 0.2171
Epoch  460  loss  0.4700532516190004 correct 50 total time 99.4061 time per epoch 0.2156
Epoch  470  loss  0.4626582179514478 correct 50 total time 100.923 time per epoch 0.2143
Epoch  480  loss  1.0885554465386535 correct 48 total time 102.4122 time per epoch 0.2129
Epoch  490  loss  1.4969180835854405 correct 49 total time 103.9005 time per epoch 0.2116
```

### GPU on Google Colab
Interestingly it is slightly slower than the CPU version, but still relatively fast
```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 150 --DATASET split --RATE 0.05 --PLOT True
```
Total time:  731.7675 Time per epoch:  1.4635
```
Epoch  0  loss  6.023488256297823 correct 33 total time 3.311 time per epoch 3.311
Epoch  10  loss  5.105850967906129 correct 42 total time 17.8701 time per epoch 1.6246
Epoch  20  loss  4.779067077542599 correct 47 total time 32.9944 time per epoch 1.5712
Epoch  30  loss  3.8415741521064084 correct 46 total time 47.41 time per epoch 1.5294
Epoch  40  loss  2.561988425960883 correct 46 total time 61.8348 time per epoch 1.5082
Epoch  50  loss  2.366843097032934 correct 49 total time 76.2543 time per epoch 1.4952
Epoch  60  loss  1.8717780381103926 correct 46 total time 90.82 time per epoch 1.4889
Epoch  70  loss  3.5613597841283067 correct 48 total time 105.8964 time per epoch 1.4915
Epoch  80  loss  2.2260890213015236 correct 49 total time 120.3108 time per epoch 1.4853
Epoch  90  loss  1.2787813150892586 correct 47 total time 134.7406 time per epoch 1.4807
Epoch  100  loss  2.267768494141068 correct 50 total time 149.1896 time per epoch 1.4771
Epoch  110  loss  1.5637103981928766 correct 48 total time 163.6739 time per epoch 1.4745
Epoch  120  loss  1.1950994222603035 correct 50 total time 178.8393 time per epoch 1.478
Epoch  130  loss  0.6561991793295731 correct 49 total time 193.3895 time per epoch 1.4763
Epoch  140  loss  1.3456009501497945 correct 50 total time 207.9258 time per epoch 1.4747
Epoch  150  loss  1.2357785357926736 correct 48 total time 222.4389 time per epoch 1.4731
Epoch  160  loss  1.2153862329168739 correct 49 total time 237.0547 time per epoch 1.4724
Epoch  170  loss  1.0644628097048034 correct 50 total time 252.084 time per epoch 1.4742
Epoch  180  loss  0.6145260795463982 correct 49 total time 266.4352 time per epoch 1.472
Epoch  190  loss  0.7845449207562192 correct 50 total time 280.8666 time per epoch 1.4705
Epoch  200  loss  1.1153232434637856 correct 50 total time 295.324 time per epoch 1.4693
Epoch  210  loss  1.2359631083377087 correct 50 total time 309.9391 time per epoch 1.4689
Epoch  220  loss  1.6299164931839791 correct 49 total time 324.9747 time per epoch 1.4705
Epoch  230  loss  1.7848908735978655 correct 49 total time 339.4115 time per epoch 1.4693
Epoch  240  loss  1.0680865921132743 correct 50 total time 353.8299 time per epoch 1.4682
Epoch  250  loss  1.1414267523147885 correct 50 total time 368.2867 time per epoch 1.4673
Epoch  260  loss  1.3098527790796626 correct 49 total time 382.764 time per epoch 1.4665
Epoch  270  loss  0.6800157509774669 correct 50 total time 397.854 time per epoch 1.4681
Epoch  280  loss  0.46316282018130817 correct 50 total time 412.3514 time per epoch 1.4674
Epoch  290  loss  0.7899781356588741 correct 50 total time 426.7959 time per epoch 1.4667
Epoch  300  loss  0.3312593616582048 correct 50 total time 441.2598 time per epoch 1.466
Epoch  310  loss  0.3057532118918508 correct 50 total time 455.7174 time per epoch 1.4653
Epoch  320  loss  0.037395163659048415 correct 49 total time 470.8508 time per epoch 1.4668
Epoch  330  loss  0.4859023559129016 correct 50 total time 485.2572 time per epoch 1.466
Epoch  340  loss  0.26974915427707363 correct 50 total time 499.7046 time per epoch 1.4654
Epoch  350  loss  0.2897305091207105 correct 50 total time 514.1057 time per epoch 1.4647
Epoch  360  loss  0.38683695911934746 correct 50 total time 528.5287 time per epoch 1.4641
Epoch  370  loss  0.5390386547789552 correct 50 total time 543.7842 time per epoch 1.4657
Epoch  380  loss  0.4287881994937994 correct 50 total time 558.193 time per epoch 1.4651
Epoch  390  loss  0.5003104793131604 correct 50 total time 572.6347 time per epoch 1.4645
Epoch  400  loss  0.21206763590609612 correct 50 total time 587.0691 time per epoch 1.464
Epoch  410  loss  0.16032245701936143 correct 50 total time 601.6933 time per epoch 1.464
Epoch  420  loss  0.23965039065651897 correct 50 total time 616.7876 time per epoch 1.4651
Epoch  430  loss  0.08144582227447479 correct 50 total time 631.2197 time per epoch 1.4645
Epoch  440  loss  0.5645245430613122 correct 49 total time 645.6056 time per epoch 1.464
Epoch  450  loss  0.21903682962601714 correct 49 total time 660.0693 time per epoch 1.4636
Epoch  460  loss  0.40815118257093363 correct 50 total time 674.6633 time per epoch 1.4635
Epoch  470  loss  0.07249051850497645 correct 50 total time 689.751 time per epoch 1.4644
Epoch  480  loss  0.06136870347944288 correct 50 total time 704.16 time per epoch 1.464
Epoch  490  loss  0.4577442034223478 correct 50 total time 718.5418 time per epoch 1.4634
```

### XOR
#### CPU
```bash
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET xor --RATE 0.05
```
Total time:  141.9181 Time per epoch:  0.2838
```
Epoch  0  loss  6.014211265902454 correct 35 total time 23.6979 time per epoch 23.6979
Epoch  10  loss  2.235478596184429 correct 48 total time 26.9746 time per epoch 2.4522
Epoch  20  loss  0.7304656302984762 correct 49 total time 29.172 time per epoch 1.3891
Epoch  30  loss  0.9872917338112743 correct 50 total time 31.2956 time per epoch 1.0095
Epoch  40  loss  0.6147755933140528 correct 50 total time 33.4298 time per epoch 0.8154
Epoch  50  loss  1.181395080984772 correct 50 total time 35.543 time per epoch 0.6969
Epoch  60  loss  0.6517027492408638 correct 50 total time 38.8247 time per epoch 0.6365
Epoch  70  loss  0.25978708778646126 correct 49 total time 40.9657 time per epoch 0.577
Epoch  80  loss  0.12964526448278269 correct 50 total time 43.0996 time per epoch 0.5321
Epoch  90  loss  0.39007204501147985 correct 50 total time 45.2444 time per epoch 0.4972
Epoch  100  loss  0.2996025032264779 correct 50 total time 47.3736 time per epoch 0.469
Epoch  110  loss  0.2132960462896781 correct 50 total time 50.647 time per epoch 0.4563
Epoch  120  loss  0.12460842265168683 correct 50 total time 52.7635 time per epoch 0.4361
Epoch  130  loss  0.05733676392908489 correct 50 total time 54.8907 time per epoch 0.419
Epoch  140  loss  0.14134789914101853 correct 50 total time 57.0168 time per epoch 0.4044
Epoch  150  loss  0.14538825274056624 correct 50 total time 59.1679 time per epoch 0.3918
Epoch  160  loss  0.050422592505111906 correct 50 total time 62.0428 time per epoch 0.3854
Epoch  170  loss  0.06618668572755369 correct 50 total time 64.5732 time per epoch 0.3776
Epoch  180  loss  0.017402431443426215 correct 50 total time 66.7087 time per epoch 0.3686
Epoch  190  loss  0.4787098305047609 correct 50 total time 68.8328 time per epoch 0.3604
Epoch  200  loss  0.15446889978696787 correct 50 total time 70.9844 time per epoch 0.3532
Epoch  210  loss  0.13940383875263965 correct 50 total time 73.3981 time per epoch 0.3479
Epoch  220  loss  0.045252718260521754 correct 50 total time 76.4022 time per epoch 0.3457
Epoch  230  loss  0.061759458217952645 correct 50 total time 78.531 time per epoch 0.34
Epoch  240  loss  0.4363821781453473 correct 50 total time 80.6499 time per epoch 0.3346
Epoch  250  loss  0.034548204145044156 correct 50 total time 82.76 time per epoch 0.3297
Epoch  260  loss  0.06022429982683163 correct 50 total time 84.894 time per epoch 0.3253
Epoch  270  loss  0.08648531909220439 correct 50 total time 88.2226 time per epoch 0.3255
Epoch  280  loss  0.37648310192705375 correct 50 total time 90.3617 time per epoch 0.3216
Epoch  290  loss  0.12845523150111415 correct 50 total time 92.5081 time per epoch 0.3179
Epoch  300  loss  0.33243045310242925 correct 50 total time 94.6343 time per epoch 0.3144
Epoch  310  loss  0.025048785936820122 correct 50 total time 96.7651 time per epoch 0.3111
Epoch  320  loss  0.33386947062846 correct 50 total time 100.1582 time per epoch 0.312
Epoch  330  loss  0.3551214658711747 correct 50 total time 102.2757 time per epoch 0.309
Epoch  340  loss  0.019270673864416568 correct 50 total time 104.4042 time per epoch 0.3062
Epoch  350  loss  0.06932918881762212 correct 50 total time 106.5653 time per epoch 0.3036
Epoch  360  loss  0.014807856108047857 correct 50 total time 108.6949 time per epoch 0.3011
Epoch  370  loss  0.26275829958473884 correct 50 total time 111.9766 time per epoch 0.3018
Epoch  380  loss  0.22904711210367226 correct 50 total time 114.1091 time per epoch 0.2995
Epoch  390  loss  0.05935460905810849 correct 50 total time 116.2514 time per epoch 0.2973
Epoch  400  loss  0.007110153167927075 correct 50 total time 118.4216 time per epoch 0.2953
Epoch  410  loss  0.07502723442289579 correct 50 total time 120.575 time per epoch 0.2934
Epoch  420  loss  0.20557468668180892 correct 50 total time 123.6413 time per epoch 0.2937
Epoch  430  loss  0.25700860027872774 correct 50 total time 126.0372 time per epoch 0.2924
Epoch  440  loss  0.043006333207681526 correct 50 total time 128.1582 time per epoch 0.2906
Epoch  450  loss  0.005731833360872772 correct 50 total time 130.2826 time per epoch 0.2889
Epoch  460  loss  0.20116603694316942 correct 50 total time 132.4481 time per epoch 0.2873
Epoch  470  loss  0.011764142114854173 correct 50 total time 135.1789 time per epoch 0.287
Epoch  480  loss  0.04020967314287059 correct 50 total time 137.8814 time per epoch 0.2867
Epoch  490  loss  0.018242664604259818 correct 50 total time 140.0047 time per epoch 0.2851
```

#### GPU
```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 200 --DATASET xor --RATE 0.05 --PLOT true
```

Total time:  753.1725 Time per epoch:  1.5063
```
Epoch  0  loss  22.08646023468951 correct 31 total time 4.1203 time per epoch 4.1203
Epoch  10  loss  5.379443390362365 correct 36 total time 19.8032 time per epoch 1.8003
Epoch  20  loss  2.6689747601216482 correct 46 total time 34.7702 time per epoch 1.6557
Epoch  30  loss  3.204928158334067 correct 47 total time 49.5661 time per epoch 1.5989
Epoch  40  loss  0.8968986597695507 correct 47 total time 64.3168 time per epoch 1.5687
Epoch  50  loss  2.6120780671615904 correct 47 total time 79.9273 time per epoch 1.5672
Epoch  60  loss  2.2834291347473705 correct 48 total time 94.8732 time per epoch 1.5553
Epoch  70  loss  0.770679725811567 correct 48 total time 109.7933 time per epoch 1.5464
Epoch  80  loss  1.453226697455118 correct 48 total time 124.698 time per epoch 1.5395
Epoch  90  loss  1.0944420151233016 correct 48 total time 140.0728 time per epoch 1.5393
Epoch  100  loss  1.2138997745712417 correct 48 total time 155.1348 time per epoch 1.536
Epoch  110  loss  1.2113520083504286 correct 48 total time 169.9253 time per epoch 1.5309
Epoch  120  loss  0.36938421772332086 correct 48 total time 184.6079 time per epoch 1.5257
Epoch  130  loss  0.9428897590780919 correct 48 total time 199.4707 time per epoch 1.5227
Epoch  140  loss  0.5657522633505205 correct 50 total time 215.048 time per epoch 1.5252
Epoch  150  loss  0.7253777730999148 correct 49 total time 229.8471 time per epoch 1.5222
Epoch  160  loss  0.7492226982299679 correct 49 total time 244.6493 time per epoch 1.5196
Epoch  170  loss  1.9167431678038498 correct 48 total time 259.4591 time per epoch 1.5173
Epoch  180  loss  1.2303242569219208 correct 49 total time 275.1255 time per epoch 1.52
Epoch  190  loss  0.30020921839410886 correct 49 total time 289.887 time per epoch 1.5177
Epoch  200  loss  0.8872360934864955 correct 49 total time 304.7211 time per epoch 1.516
Epoch  210  loss  0.22295819317048562 correct 49 total time 319.6433 time per epoch 1.5149
Epoch  220  loss  0.9986995360263082 correct 49 total time 335.4064 time per epoch 1.5177
Epoch  230  loss  0.18894532107997603 correct 50 total time 350.4104 time per epoch 1.5169
Epoch  240  loss  0.9660175541296827 correct 49 total time 365.2228 time per epoch 1.5154
Epoch  250  loss  0.36383002041793755 correct 49 total time 380.0302 time per epoch 1.5141
Epoch  260  loss  1.3426039565746235 correct 49 total time 395.0697 time per epoch 1.5137
Epoch  270  loss  0.748329707586959 correct 49 total time 410.33 time per epoch 1.5141
Epoch  280  loss  0.500065874684057 correct 50 total time 425.1115 time per epoch 1.5129
Epoch  290  loss  0.10469465914188039 correct 49 total time 439.8996 time per epoch 1.5117
Epoch  300  loss  0.14577547909451238 correct 49 total time 454.7156 time per epoch 1.5107
Epoch  310  loss  0.29584277665596675 correct 49 total time 470.3498 time per epoch 1.5124
Epoch  320  loss  0.41370484630656346 correct 49 total time 485.087 time per epoch 1.5112
Epoch  330  loss  0.23853976609965166 correct 50 total time 499.8955 time per epoch 1.5103
Epoch  340  loss  0.10120978967001434 correct 49 total time 514.7502 time per epoch 1.5095
Epoch  350  loss  0.9311390082402182 correct 49 total time 530.329 time per epoch 1.5109
Epoch  360  loss  1.4049963852481855 correct 49 total time 545.1037 time per epoch 1.51
Epoch  370  loss  0.14943313969074623 correct 49 total time 559.9393 time per epoch 1.5093
Epoch  380  loss  0.9651727257622114 correct 49 total time 574.745 time per epoch 1.5085
Epoch  390  loss  0.4225802427035762 correct 50 total time 589.9555 time per epoch 1.5088
Epoch  400  loss  0.877091291025727 correct 49 total time 605.0865 time per epoch 1.5089
Epoch  410  loss  0.9545517551007882 correct 49 total time 619.8395 time per epoch 1.5081
Epoch  420  loss  1.0932987842147273 correct 49 total time 634.6289 time per epoch 1.5074
Epoch  430  loss  0.146409291810711 correct 49 total time 649.4709 time per epoch 1.5069
Epoch  440  loss  0.12746090616894865 correct 50 total time 665.0734 time per epoch 1.5081
Epoch  450  loss  0.4331359570603037 correct 50 total time 679.9461 time per epoch 1.5076
Epoch  460  loss  0.13195215647488515 correct 49 total time 694.7498 time per epoch 1.507
Epoch  470  loss  0.13330937327180825 correct 50 total time 709.5192 time per epoch 1.5064
Epoch  480  loss  0.0776658081820214 correct 49 total time 725.0229 time per epoch 1.5073
Epoch  490  loss  0.08333898113631964 correct 50 total time 739.7827 time per epoch 1.5067
````