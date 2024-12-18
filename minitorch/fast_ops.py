from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

from numba import njit as _njit, prange
import numpy as np

from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
    MAX_DIMS,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides, Index


# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Decorator for JIT compiling functions with NUMBA."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# inv = njit(inv)
# inv_back  = njit(inv_back)

to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        # ASSIGN2.2:
        if (
            len(out_strides) != len(in_strides)
            or (out_strides != in_strides).any()
            or (out_shape != in_shape).any()
        ):
            for i in prange(len(out)):
                out_index: Index = np.zeros(MAX_DIMS, np.int16)  # type: ignore
                in_index: Index = np.zeros(MAX_DIMS, np.int16)  # type: ignore
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                o = index_to_position(out_index, out_strides)
                j = index_to_position(in_index, in_strides)
                out[o] = fn(in_storage[j])
        else:
            # When `out` and `in` are stride-aligned, avoid indexing
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
        # END ASSIGN2.2

    return njit(_map, parallel=True)  # type: ignorep


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        if (
            len(out_strides) != len(a_strides)
            or len(out_strides) != len(b_strides)
            or (out_strides != a_strides).any()
            or (out_strides != b_strides).any()
            or (out_shape != a_shape).any()
            or (out_shape != b_shape).any()
        ):
            for i in prange(len(out)):
                out_index: Index = np.zeros(MAX_DIMS, np.int32)
                a_index: Index = np.zeros(MAX_DIMS, np.int32)
                b_index: Index = np.zeros(MAX_DIMS, np.int32)
                to_index(i, out_shape, out_index)
                o = index_to_position(out_index, out_strides)
                broadcast_index(out_index, out_shape, a_shape, a_index)
                j = index_to_position(a_index, a_strides)
                broadcast_index(out_index, out_shape, b_shape, b_index)
                k = index_to_position(b_index, b_strides)
                out[o] = fn(a_storage[j], b_storage[k])
        else:
            # When out, a, b are stride-aligned, avoid indexing
            for i in prange(len(out)):
                out[i] = fn(a_storage[i], b_storage[i])

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        for i in prange(len(out)):
            out_index = np.empty(MAX_DIMS, np.int32)
            size = a_shape[reduce_dim]  # the reduce size
            # get the index of i
            to_index(i, out_shape, out_index)
            # get the position
            o = index_to_position(out_index, out_strides)
            j = index_to_position(out_index, a_strides)
            # the accumulation
            a = out[o]
            step = a_strides[reduce_dim]
            for s in range(size):
                a = fn(a, a_storage[j])
                j += step
            out[o] = a

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    total_elements = out_shape[0] * out_shape[1] * out_shape[2]

    for idx in prange(total_elements):  # Parallel over all output elements
        # Calculate batch, row, and column indices
        batch = idx // (out_shape[1] * out_shape[2])
        idx_in_batch = idx % (out_shape[1] * out_shape[2])
        row = idx_in_batch // out_shape[2]
        col = idx_in_batch % out_shape[2]

        # Calculate starting positions using precomputed batch strides
        a_pos = batch * a_batch_stride + row * a_strides[1]
        b_pos = batch * b_batch_stride + col * b_strides[2]

        # Compute the dot product
        result = 0.0
        for k in range(a_shape[2]):  # Shared dimension
            result += (
                a_storage[a_pos + k * a_strides[2]]
                * b_storage[b_pos + k * b_strides[1]]
            )

        # Store the result
        out_pos = batch * out_strides[0] + row * out_strides[1] + col * out_strides[2]
        out[out_pos] = result
    # Parallel loop through batches and matrix dimensions
    # for batch in prange(out_shape[0]):  # Batch dimension
    #     for row in prange(out_shape[1]):  # Output matrix rows
    #         for col in prange(out_shape[2]):  # Output matrix columns
    #             # Calculate starting positions in A and B
    #             a_pos = (
    #                 batch * a_batch_stride + row * a_strides[1]
    #             )  # Position in matrix A
    #             b_pos = (
    #                 batch * b_batch_stride + col * b_strides[2]
    #             )  # Position in matrix B

    #             # Do matrix multiplication for this position
    #             result = 0.0
    #             for k in range(a_shape[2]):  # Inner product dimension
    #                 # A[batch, row, k] * B[batch, k, col]
    #                 result += a_storage[a_pos] * b_storage[b_pos]
    #                 # Move to next element in row/column
    #                 a_pos += a_strides[2]  # Move along row in A
    #                 b_pos += b_strides[1]  # Move down column in B

    #             # Store result in output
    #             out_pos = (
    #                 batch * out_strides[0] + row * out_strides[1] + col * out_strides[2]
    #             )
    #             out[out_pos] = result

    # A:
    # [
    #     [1 ,2 ]
    #     [3, 4]
    # ]
    # B:
    # [
    #     [5, 6]
    #     [7, 8]
    # ]
    # A @ B:
    # [
    #     [1*5 + 2*7, 1*6 + 2*8]
    #     [3*5 + 4*7, 3*6 + 4*8]
    # ]
    # spelled out in words
    # A @ B:
    # [
    #     [ A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0], A[0, 0] * B[0, 1] + A[0, 1] * B[1, 1] ]
    #     [ A[1, 0] * B[0, 0] + A[1, 1] * B[1, 0], A[1, 0] * B[0, 1] + A[1, 1] * B[1, 1] ]
    # ]
    # on a basic level with for loops
    # for row_a in range(A.rows):
    #     for col_b in range(B.cols):
    #         accumulator = 0
    #         for col_a in range(A.cols):
    #              accumulator += A[row_a, col_a] * B[col_a, col_b]
    #         out[row_a, col_b] = accumulator
    # since we start looping through out storage, we need to basically work backwards from storage position to the index out[i, j] or something?
    # for out_ordinal in prange(len(out)):
    #     # find index in out
    #     out_index = np.zeros(2, np.int32)
    #     to_index(out_ordinal, out_shape, out_index)

    #     #  out index
    #     row_a, col_b = out_index[0], out_index[1]

    #     # accumulator
    #     accumulator = 0
    #     a_index = np.array([row_a, 0], np.int32)
    #     b_index = np.array([0, col_b], np.int32)
    #     for inner in range(a_shape[-1]):
    #         a_index[1] = inner
    #         b_index[0] = inner
    #         a_ordinal = index_to_position(a_index, a_strides)
    #         b_ordinal = index_to_position(b_index, b_strides)
    #         accumulator += a_storage[a_ordinal] * b_storage[b_ordinal]
    #     out[out_ordinal] = accumulator


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
