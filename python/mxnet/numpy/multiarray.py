#!/usr/bin/env python

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=too-many-lines, unused-argument
"""numpy ndarray and util functions."""

from __future__ import absolute_import
from __future__ import division

try:
    from __builtin__ import slice as py_slice
except ImportError:
    from builtins import slice as py_slice

from array import array as native_array
import sys
import ctypes
import warnings
import numpy as _np
from ..ndarray import NDArray, _DTYPE_NP_TO_MX, _GRAD_REQ_MAP
from ..ndarray import indexing_key_expand_implicit_axes, get_indexing_dispatch_code,\
                      get_oshape_of_gather_nd_op
from ..ndarray._internal import _set_np_ndarray_class
from . import _op as _mx_np_op
from ..base import check_call, _LIB, NDArrayHandle
from ..base import mx_real_t, c_array_buf, mx_uint, numeric_types, integer_types
from ..context import Context
from ..util import _sanity_check_params, set_module
from ..context import current_context
from ..ndarray import numpy as _mx_nd_np
from ..ndarray.numpy import _internal as _npi

__all__ = ['ndarray', 'empty', 'array', 'zeros', 'ones', 'full', 'add', 'subtract', 'multiply', 'divide',
           'mod', 'power', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'log10', 'sqrt', 'cbrt',
           'abs', 'absolute', 'exp', 'expm1', 'arcsin', 'arccos', 'arctan', 'sign', 'log',
           'degrees', 'log2', 'log1p', 'rint', 'radians', 'reciprocal', 'square', 'negative',
           'fix', 'ceil', 'floor', 'trunc', 'logical_not', 'arcsinh', 'arccosh', 'arctanh',
           'tensordot', 'linspace', 'expand_dims', 'tile', 'arange', 'split', 'concatenate',
           'stack']

# Return code for dispatching indexing function call
_NDARRAY_UNSUPPORTED_INDEXING = -1
_NDARRAY_BASIC_INDEXING = 0
_NDARRAY_ADVANCED_INDEXING = 1

# This function is copied from ndarray.py since pylint
# keeps giving false alarm error of undefined-all-variable
def _new_alloc_handle(shape, ctx, delay_alloc, dtype=mx_real_t):
    """Return a new handle with specified shape and context.

    Empty handle is only used to hold results.

    Returns
    -------
    handle
        A new empty `ndarray` handle.
    """
    hdl = NDArrayHandle()
    check_call(_LIB.MXNDArrayCreateEx(
        c_array_buf(mx_uint, native_array('I', shape)),
        mx_uint(len(shape)),
        ctypes.c_int(ctx.device_typeid),
        ctypes.c_int(ctx.device_id),
        ctypes.c_int(int(delay_alloc)),
        ctypes.c_int(int(_DTYPE_NP_TO_MX[_np.dtype(dtype).type])),
        ctypes.byref(hdl)))
    return hdl


# Have to use 0 as default value for stype since pylint does not allow
# importing _STORAGE_TYPE_DEFAULT from ndarray.py.
def _np_ndarray_cls(handle, writable=True, stype=0):
    if stype != 0:
        raise ValueError('_np_ndarray_cls currently only supports default storage '
                         'type, while received stype = {}'.format(stype))
    return ndarray(handle, writable=writable)


_set_np_ndarray_class(_np_ndarray_cls)


def _get_index(idx):
    if isinstance(idx, NDArray) and not isinstance(idx, ndarray):
        raise TypeError('Cannot have mx.nd.NDArray as index')
    if isinstance(idx, ndarray):
        return idx.as_nd_ndarray()
    elif sys.version_info[0] > 2 and isinstance(idx, range):
        return array(_np.arange(idx.start, idx.stop, idx.step, dtype=_np.int32)).as_nd_ndarray()
    else:
        return idx


@set_module('mxnet.numpy')  # pylint: disable=invalid-name
class ndarray(NDArray):
    """
    An array object represents a multidimensional, homogeneous array of fixed-size items.
    An associated data-type object describes the format of each element in the array
    (its byte-order, how many bytes it occupies in memory, whether it is an integer, a
    floating point number, or something else, etc.). Arrays should be constructed using
    `array`, `zeros` or `empty`. Currently, only c-contiguous arrays are supported.
    """
    def _get_np_basic_indexing(self, key):
        """
        This function indexes ``self`` with a tuple of `slice` objects only.
        """
        key_nd = tuple(idx for idx in key if idx is not None)
        if len(key_nd) < self.ndim:
            raise RuntimeError(
                'too few indices after normalization: expected `ndim` ({}) '
                'but got {}. This is a bug, please report it!'
                ''.format(self.ndim, len(key_nd))
            )
        if len(key_nd) > self.ndim:
            raise IndexError(
                'too many indices ({}) for array with {} dimensions'
                ''.format(len(key_nd), self.ndim)
            )

        none_axes = [ax for ax in range(len(key)) if key[ax] is None]  # pylint: disable=invalid-name
        slc_key, int_axes = self._basic_indexing_key_int_to_slice(key_nd)
        new_axes = self._new_axes_after_basic_indexing(none_axes, key)

        # Check bounds for integer axes
        for ax in int_axes:  # pylint: disable=invalid-name
            if not -self.shape[ax] <= key_nd[ax] < self.shape[ax]:
                raise IndexError(
                    'index {} is out of bounds for axis {} with size {}'
                    ''.format(key_nd[ax], ax, self.shape[ax]))

        if self._basic_indexing_slice_is_contiguous(slc_key, self.shape):
            # Create a shared-memory view by using low-level flat slicing
            flat_begin, flat_end = self._basic_indexing_contiguous_flat_begin_end(
                slc_key, self.shape
            )
            handle = NDArrayHandle()
            flat_self = self.reshape_view(-1)
            check_call(
                _LIB.MXNDArraySlice(
                    flat_self.handle,
                    mx_uint(flat_begin),
                    mx_uint(flat_end),
                    ctypes.byref(handle),
                )
            )
            sliced_shape = self._basic_indexing_sliced_shape(slc_key, self.shape)
            sliced = self.__class__(handle=handle, writable=self.writable)
            if 0 in sliced_shape:
                sliced = sliced.reshape(sliced_shape)
            else:
                sliced = sliced.reshape_view(sliced_shape)

        else:
            begin, end, step = self._basic_indexing_key_to_begin_end_step(
                slc_key, self.shape, keep_none=True
            )
            sliced = _npi.slice(self, begin, end, step)

        # Reshape to final shape due to integer and `None` entries in `key`.
        final_shape = [sliced.shape[i] for i in range(sliced.ndim) if i not in int_axes]
        for ax in new_axes:  # pylint: disable=invalid-name
            final_shape.insert(ax, 1)

        if sliced.size == 0:
            return sliced.reshape(tuple(final_shape))
        else:
            return sliced.reshape_view(tuple(final_shape))

    def _get_np_advanced_indexing(self, key):
        idcs, new_axes = self._get_index_nd(key)
        if type(idcs) == NDArray:  # pylint: disable=unidiomatic-typecheck
            idcs = idcs.as_np_ndarray()
        else:
            idcs = _npi.stack(*[i if isinstance(i, self.__class__) else i.as_np_ndarray() for i in idcs])
        sliced = _npi.gather_nd(self, idcs)
        # Reshape due to `None` entries in `key`.
        if new_axes:
            final_shape = [sliced.shape[i] for i in range(sliced.ndim)]
            for ax in new_axes:  # pylint: disable=invalid-name
                final_shape.insert(ax, 1)
            return sliced.reshape(tuple(final_shape))
        else:
            return sliced

    def _set_np_advanced_indexing(self, key, value):
        """This function is called by __setitem__ when key is an advanced index."""
        idcs, new_axes = self._get_index_nd(key)
        if type(idcs) == NDArray:  # pylint: disable=unidiomatic-typecheck
            idcs = idcs.as_np_ndarray()
        else:
            idcs = _npi.stack(*[i if isinstance(i, self.__class__) else i.as_np_ndarray() for i in idcs])
        vshape = get_oshape_of_gather_nd_op(self.shape, idcs.shape)
        value_nd = self._prepare_value_nd(value, bcast_shape=vshape, squeeze_axes=new_axes)
        self._scatter_set_nd(value_nd, idcs)

    # pylint: disable=too-many-return-statements
    def __getitem__(self, key):
        """
        Overriding the method in NDArray class in a numpy fashion.
        Calling numpy ndarray's _get_np_basic_indexing(key) and _get_np_advanced_indexing(key).
        """
        ndim = self.ndim
        shape = self.shape
        if ndim == 0:
            if key != ():
                raise IndexError('scalar tensor can only accept `()` as index')
        # Handle simple cases for higher speed
        if isinstance(key, tuple) and len(key) == 0:
            return self
        if isinstance(key, tuple) and len(key) == ndim\
                and all(isinstance(idx, integer_types) for idx in key):
            out = self
            for idx in key:
                out = out[idx]
            return out
        if isinstance(key, integer_types):
            if key > shape[0] - 1:
                raise IndexError(
                    'index {} is out of bounds for axis 0 with size {}'.format(
                        key, shape[0]))
            return self._at(key)
        elif isinstance(key, py_slice):
            if (key.step is None or key.step == 1):
                if  key.start is not None or key.stop is not None:
                    return self._slice(key.start, key.stop)
                else:
                    return self
            elif key.step == 0:
                raise ValueError("slice step cannot be zero")

        key = indexing_key_expand_implicit_axes(key, self.shape)
        indexing_dispatch_code = get_indexing_dispatch_code(key)
        if indexing_dispatch_code == _NDARRAY_BASIC_INDEXING:
            return self._get_np_basic_indexing(key)
        elif indexing_dispatch_code == _NDARRAY_ADVANCED_INDEXING:
            return self._get_np_advanced_indexing(key)
        else:
            raise RuntimeError

    def __setitem__(self, key, value):
        """
        x.__setitem__(i, y) <=> x[i]=y
        Sets ``self[key]`` to ``value``.

        Overriding the method in NDArray class in a numpy fashion.
        """
        if isinstance(value, NDArray) and not isinstance(value, ndarray):
            raise TypeError('Cannot assign mx.nd.NDArray to mxnet.numpy.ndarray')
        if self.ndim == 0:
            if not isinstance(key, tuple) or len(key) != 0:
                raise IndexError('scalar tensor can only accept `()` as index')
            if isinstance(value, numeric_types):
                self.full(value)
            elif isinstance(value, ndarray) and value.size == 1:
                if value.shape != self.shape:
                    value = value.reshape(self.shape)
                value.copyto(self)
            elif isinstance(value, (_np.ndarray, _np.generic)) and value.size == 1:
                if isinstance(value, _np.generic) or value.shape != self.shape:
                    value = value.reshape(self.shape)
                self._sync_copyfrom(value)
            else:
                raise ValueError('setting an array element with a sequence.')
        else:
            key = indexing_key_expand_implicit_axes(key, self.shape)
            slc_key = tuple(idx for idx in key if idx is not None)
            if len(slc_key) < self.ndim:
                raise RuntimeError(
                    'too few indices after normalization: expected `ndim` ({}) '
                    'but got {}. This is a bug, please report it!'
                    ''.format(self.ndim, len(slc_key))
                )
            if len(slc_key) > self.ndim and self.ndim != 0:
                raise IndexError(
                    'too many indices ({}) for array with {} dimensions'
                    ''.format(len(slc_key), self.ndim)
                )
            indexing_dispatch_code = get_indexing_dispatch_code(slc_key)
            if indexing_dispatch_code == _NDARRAY_BASIC_INDEXING:
                self._set_nd_basic_indexing(key, value)  # function is inheritated from NDArray class
            elif indexing_dispatch_code == _NDARRAY_ADVANCED_INDEXING:
                self._set_np_advanced_indexing(key, value)
            else:
                raise ValueError(
                    'Indexing NDArray with index {} of type {} is not supported'
                    ''.format(key, type(key))
                )

    def _prepare_value_nd(self, value, bcast_shape, squeeze_axes=None):
        """Return a broadcast `ndarray` with same context and dtype as ``self``.
        For setting item, The returned `ndarray` is squeezed according to squeeze_axes since the
        value_nd is assigned to not yet expanded space in original array.
        `value`: numeric types or array like.
        `bcast_shape`: a shape tuple.
        `squeeze_axes`: a sequence of axes to squeeze in the value array.
        Note: mxnet.numpy.ndarray not support NDArray as assigned value.
        """
        if isinstance(value, numeric_types):
            value_nd = full(bcast_shape, value, ctx=self.context, dtype=self.dtype)
        elif isinstance(value, self.__class__):
            value_nd = value.as_in_context(self.context)
            if value_nd.dtype != self.dtype:
                value_nd = value_nd.astype(self.dtype)
        else:
            try:
                value_nd = array(value, ctx=self.context, dtype=self.dtype)
            except:
                raise TypeError('mxnet.np.ndarray does not support assignment with non-array-like '
                                'object {} of type {}'.format(value, type(value)))

        # For advanced indexing setitem, if there is None in indices, we need to squeeze the
        # assigned value_nd since None is also ignored in slicing the original array.
        if squeeze_axes and value_nd.ndim > len(bcast_shape):
            squeeze_axes = tuple([ax for ax in squeeze_axes if ax < len(value_nd.shape)])
            value_nd = value_nd.squeeze(axis=tuple(squeeze_axes))

        if value_nd.shape != bcast_shape:
            if value_nd.size == 0:
                value_nd = value_nd.reshape(bcast_shape)
            else:
                value_nd = value_nd.broadcast_to(bcast_shape)
        return value_nd


    def __add__(self, other):
        """x.__add__(y) <=> x + y"""
        return add(self, other)

    def __iadd__(self, other):
        """x.__iadd__(y) <=> x += y"""
        if not self.writable:
            raise ValueError('trying to add to a readonly ndarray')
        return add(self, other, out=self)

    def __sub__(self, other):
        """x.__sub__(y) <=> x - y"""
        return subtract(self, other)

    def __isub__(self, other):
        """x.__isub__(y) <=> x -= y"""
        if not self.writable:
            raise ValueError('trying to subtract from a readonly ndarray')
        return subtract(self, other, out=self)

    def __rsub__(self, other):
        """x.__rsub__(y) <=> y - x"""
        return subtract(other, self)

    def __mul__(self, other):
        """x.__mul__(y) <=> x * y"""
        return multiply(self, other)

    def __neg__(self):
        return self.__mul__(-1.0)

    def __imul__(self, other):
        """x.__imul__(y) <=> x *= y"""
        if not self.writable:
            raise ValueError('trying to add to a readonly ndarray')
        return multiply(self, other, out=self)

    def __rmul__(self, other):
        """x.__rmul__(y) <=> y * x"""
        return self.__mul__(other)

    def __div__(self, other):
        raise AttributeError('ndarray.__div__ is replaced by __truediv__. If you are using'
                             ' Python2, please use the statement from __future__ import division'
                             ' to change the / operator to mean true division throughout the'
                             ' module. If you are using Python3, this error should not have'
                             ' been encountered.')

    def __rdiv__(self, other):
        raise AttributeError('ndarray.__rdiv__ is replaced by __rtruediv__. If you are using'
                             ' Python2, please use the statement from __future__ import division'
                             ' to change the / operator to mean true division throughout the'
                             ' module. If you are using Python3, this error should not have'
                             ' been encountered.')

    def __idiv__(self, other):
        raise AttributeError('ndarray.__idiv__ is replaced by __irtruediv__. If you are using'
                             ' Python2, please use the statement from __future__ import division'
                             ' to change the / operator to mean true division throughout the'
                             ' module. If you are using Python3, this error should not have'
                             ' been encountered.')

    def __truediv__(self, other):
        """x.__truediv__(y) <=> x / y"""
        return divide(self, other)

    def __rtruediv__(self, other):
        """x.__rtruediv__(y) <=> y / x"""
        return divide(other, self)

    def __itruediv__(self, other):
        return divide(self, other, out=self)

    def __mod__(self, other):
        """x.__mod__(y) <=> x % y"""
        return mod(self, other)

    def __rmod__(self, other):
        """x.__rmod__(y) <=> y % x"""
        return mod(other, self)

    def __imod__(self, other):
        """x.__imod__(y) <=> x %= y"""
        return mod(self, other, out=self)

    def __pow__(self, other):
        """x.__pow__(y) <=> x ** y"""
        return power(self, other)

    def __rpow__(self, other):
        """x.__rpow__(y) <=> y ** x"""
        return power(other, self)

    def __eq__(self, other):
        """x.__eq__(y) <=> x == y"""
        # TODO(junwu): Return boolean ndarray when dtype=bool_ is supported
        if isinstance(other, ndarray):
            return _npi.equal(self, other)
        elif isinstance(other, numeric_types):
            return _npi.equal_scalar(self, float(other))
        else:
            raise TypeError("ndarray does not support type {} as operand".format(str(type(other))))

    def __hash__(self):
        raise NotImplementedError

    def __ne__(self, other):
        """x.__ne__(y) <=> x != y"""
        # TODO(junwu): Return boolean ndarray when dtype=bool_ is supported
        if isinstance(other, ndarray):
            return _npi.not_equal(self, other)
        elif isinstance(other, numeric_types):
            return _npi.not_equal_scalar(self, float(other))
        else:
            raise TypeError("ndarray does not support type {} as operand".format(str(type(other))))

    def __gt__(self, other):
        """x.__gt__(y) <=> x > y"""
        # TODO(junwu): Return boolean ndarray when dtype=bool_ is supported
        if isinstance(other, ndarray):
            return _npi.greater(self, other)
        elif isinstance(other, numeric_types):
            return _npi.greater_scalar(self, float(other))
        else:
            raise TypeError("ndarray does not support type {} as operand".format(str(type(other))))

    def __ge__(self, other):
        """x.__ge__(y) <=> x >= y"""
        # TODO(junwu): Return boolean ndarray when dtype=bool_ is supported
        if isinstance(other, ndarray):
            return _npi.greater_equal(self, other)
        elif isinstance(other, numeric_types):
            return _npi.greater_equal_scalar(self, float(other))
        else:
            raise TypeError("ndarray does not support type {} as operand".format(str(type(other))))

    def __lt__(self, other):
        """x.__lt__(y) <=> x < y"""
        # TODO(junwu): Return boolean ndarray when dtype=bool_ is supported
        if isinstance(other, ndarray):
            return _npi.less(self, other)
        elif isinstance(other, numeric_types):
            return _npi.less_scalar(self, float(other))
        else:
            raise TypeError("ndarray does not support type {} as operand".format(str(type(other))))

    def __le__(self, other):
        """x.__le__(y) <=> x <= y"""
        # TODO(junwu): Return boolean ndarray when dtype=bool_ is supported
        if isinstance(other, ndarray):
            return _npi.less_equal(self, other)
        elif isinstance(other, numeric_types):
            return _npi.less_equal_scalar(self, float(other))
        else:
            raise TypeError("ndarray does not support type {} as operand".format(str(type(other))))

    def __bool__(self):
        num_elements = self.size
        if num_elements == 0:
            warnings.simplefilter('default')
            warnings.warn('The truth value of an empty array is ambiguous. Returning False, but in'
                          ' future this will result in an error.', DeprecationWarning)
            return False
        elif num_elements == 1:
            return bool(self.item())
        else:
            raise ValueError("The truth value of an ndarray with multiple elements is ambiguous.")

    __nonzero__ = __bool__

    def __float__(self):
        num_elements = self.size
        if num_elements != 1:
            raise TypeError('only size-1 arrays can be converted to Python scalars')
        return float(self.item())

    def __int__(self):
        num_elements = self.size
        if num_elements != 1:
            raise TypeError('only size-1 arrays can be converted to Python scalars')
        return int(self.item())

    def __len__(self):
        """Number of elements along the first axis."""
        shape = self.shape
        if len(shape) == 0:
            raise TypeError('len() of unsized object')
        return self.shape[0]

    def __reduce__(self):
        return ndarray, (None,), self.__getstate__()

    def item(self, *args):
        """Copy an element of an array to a standard Python scalar and return it.

        Parameters
        ----------
        *args : Arguments (variable number and type)
            none: in this case, the method only works for arrays with one element (a.size == 1),
            which element is copied into a standard Python scalar object and returned.

            int_type: this argument is interpreted as a flat index into the array, specifying which
            element to copy and return.

            tuple of int_types: functions as does a single int_type argument, except that the
            argument is interpreted as an nd-index into the array.

        Returns
        -------
        z : Standard Python scalar object
            A copy of the specified element of the array as a suitable Python scalar.
        """
        # TODO(junwu): no need to call asnumpy() on the whole array.
        return self.asnumpy().item(*args)

    @property
    # pylint: disable= invalid-name, undefined-variable
    def T(self):
        """Same as self.transpose(). This always returns a copy of self."""
        return self.transpose()
    # pylint: enable= invalid-name, undefined-variable

    def all(self, axis=None, out=None, keepdims=False):
        raise NotImplementedError

    def any(self, axis=None, out=None, keepdims=False):
        raise NotImplementedError

    def as_nd_ndarray(self):
        """Convert mxnet.numpy.ndarray to mxnet.ndarray.NDArray to use its fluent methods."""
        hdl = NDArrayHandle()
        check_call(_LIB.MXShallowCopyNDArray(self.handle, ctypes.byref(hdl)))
        return NDArray(handle=hdl, writable=self.writable)

    def as_np_ndarray(self):
        """A convenience function for creating a numpy ndarray from the current ndarray
        with zero copy. For this class, it just returns itself since it's already a
        numpy ndarray."""
        return self

    def __repr__(self):
        """
        Returns a string representation of the array. The dtype of the ndarray will not
        be appended to the string if it is `float32`. The context of the ndarray will
        be appended for devices other than CPU.

        Examples
        --------
        >>> from mxnet import np, npx
        >>> a = np.random.uniform(size=(2, 3))
        >>> a
        array([[0.5488135 , 0.5928446 , 0.71518934],
               [0.84426576, 0.60276335, 0.8579456 ]])
        >>> print(a)
        [[0.5488135  0.5928446  0.71518934]
         [0.84426576 0.60276335 0.8579456 ]]
        >>> a.dtype
        <class 'numpy.float32'>
        >>> b = a.astype(np.float64)
        >>> b
        array([[0.54881352, 0.59284461, 0.71518934],
               [0.84426576, 0.60276335, 0.85794562]], dtype=float64)
        >>> print(b)
        [[0.54881352 0.59284461 0.71518934]
         [0.84426576 0.60276335 0.85794562]]
        >>> b.dtype
        <class 'numpy.float64'>
        >>> c = a.copyto(npx.gpu(0))
        >>> c
        array([[0.5488135 , 0.5928446 , 0.71518934],
               [0.84426576, 0.60276335, 0.8579456 ]], ctx=gpu(0))
        >>> print(c)
        [[0.5488135  0.5928446  0.71518934]
         [0.84426576 0.60276335 0.8579456 ]] @gpu(0)
        >>> d = b.copyto(npx.gpu(0))
        >>> d
        array([[0.54881352, 0.59284461, 0.71518934],
               [0.84426576, 0.60276335, 0.85794562]], dtype=float64, ctx=gpu(0))
        >>> print(d)
        [[0.54881352 0.59284461 0.71518934]
         [0.84426576 0.60276335 0.85794562]] @gpu(0)
        """
        array_str = self.asnumpy().__repr__()
        dtype = self.dtype
        if 'dtype=' in array_str:
            if dtype == _np.float32:
                array_str = array_str[:array_str.rindex(',')] + ')'
        elif dtype != _np.float32:
            array_str = array_str[:-1] + ', dtype={})'.format(dtype.__name__)

        context = self.context
        if context.device_type == 'cpu':
            return array_str
        return array_str[:-1] + ', ctx={})'.format(str(context))

    def __str__(self):
        """Returns a string representation of the array."""
        array_str = self.asnumpy().__str__()
        context = self.context
        if context.device_type == 'cpu' or self.ndim == 0:
            return array_str
        return '{array} @{ctx}'.format(array=array_str, ctx=context)

    def attach_grad(self, grad_req='write'):  # pylint: disable=arguments-differ
        """Attach a gradient buffer to this ndarray, so that `backward`
        can compute gradient with respect to it.

        Parameters
        ----------
        grad_req : {'write', 'add', 'null'}
            How gradient will be accumulated.
            - 'write': gradient will be overwritten on every backward.
            - 'add': gradient will be added to existing value on every backward.
            - 'null': do not compute gradient for this NDArray.
        """
        grad = _mx_np_op.zeros_like(self)  # pylint: disable=undefined-variable
        grad_req = _GRAD_REQ_MAP[grad_req]
        check_call(_LIB.MXAutogradMarkVariables(
            1, ctypes.pointer(self.handle),
            ctypes.pointer(mx_uint(grad_req)),
            ctypes.pointer(grad.handle)))

    @property
    def grad(self):
        """Returns gradient buffer attached to this ndarray."""
        hdl = NDArrayHandle()
        check_call(_LIB.MXNDArrayGetGrad(self.handle, ctypes.byref(hdl)))
        if hdl.value is None:
            return None
        return _np_ndarray_cls(hdl)

    def detach(self):
        """Returns a new ndarray, detached from the current graph."""
        hdl = NDArrayHandle()
        check_call(_LIB.MXNDArrayDetach(self.handle, ctypes.byref(hdl)))
        return _np_ndarray_cls(hdl)

    def astype(self, dtype, *args, **kwargs):  # pylint: disable=arguments-differ,unused-argument
        """
        Copy of the array, cast to a specified type.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, optional
            Default `True`. By default, astype always returns a newly
            allocated ndarray on the same context. If this is set to
            `False`, and the dtype requested is the same as the ndarray's
            dtype, the ndarray is returned instead of a copy.

        Returns
        -------
        arr_t : ndarray
            Unless `copy` is False and the other conditions for returning the input
            array are satisfied (see description for `copy` input parameter), `arr_t`
            is a new array of the same shape as the input array with `dtype`.
        """
        _sanity_check_params('astype', ['order', 'casting', 'subok'], kwargs)
        copy = kwargs.get('copy', True)
        if not copy and _np.dtype(dtype) == self.dtype:
            return self

        res = empty(self.shape, dtype=dtype, ctx=self.context)
        self.copyto(res)
        return res

    def copyto(self, other):
        """Copies the value of this array to another array.

        If ``other`` is a ``ndarray`` object, then ``other.shape`` and
        ``self.shape`` should be the same. This function copies the value from
        ``self`` to ``other``.

        If ``other`` is a context, a new ``np.ndarray`` will be first created on
        the target context, and the value of ``self`` is copied.

        Parameters
        ----------
        other : ndarray or Context
            The destination array or context.

        Returns
        -------
        out: ndarray
            The copied array. If ``other`` is an ``ndarray``, then the return value
            and ``other`` will point to the same ``ndarray``.

        Examples
        --------
        >>> x = np.ones((2,3))
        >>> y = np.zeros((2,3), ctx=mx.gpu(0))
        >>> z = x.copyto(y)
        >>> z is y
        True
        >>> y.asnumpy()
        array([[ 1.,  1.,  1.],
               [ 1.,  1.,  1.]], dtype=float32)
        """
        if isinstance(other, ndarray):
            if other.handle is self.handle:
                warnings.warn('You are attempting to copy an array to itself', RuntimeWarning)
                return False
            return _npi.copyto(self, out=other)
        elif isinstance(other, Context):
            hret = ndarray(_new_alloc_handle(self.shape, other, True, self.dtype))
            return _npi.copyto(self, out=hret)
        else:
            raise TypeError('copyto does not support type ' + str(type(other)))

    def asscalar(self):
        raise AttributeError('mxnet.numpy.ndarray object has no attribute asscalar')

    def argmax(self, axis=None, out=None):  # pylint: disable=arguments-differ
        raise NotImplementedError

    def as_in_context(self, context):
        """Returns an array on the target device with the same value as this array.

        If the target context is the same as ``self.context``, then ``self`` is
        returned.  Otherwise, a copy is made.

        Parameters
        ----------
        context : Context
            The target context.

        Returns
        -------
        ndarray
            The target array.
        """
        if self.context == context:
            return self
        return self.copyto(context)

    def copy(self, order='C'):  # pylint: disable=arguments-differ
        if order != 'C':
            raise NotImplementedError('ndarray.copy only supports order=\'C\', while '
                                      'received {}'.format(str(order)))
        return super(ndarray, self).copy().as_np_ndarray()

    def dot(self, b, out=None):
        raise NotImplementedError

    def reshape(self, *args, **kwargs):  # pylint: disable=arguments-differ
        """Returns an array containing the same data with a new shape.

        Notes
        -----
        Unlike the free function `numpy.reshape`, this method on `ndarray` allows
        the elements of the shape parameter to be passed in as separate arguments.
        For example, ``a.reshape(10, 11)`` is equivalent to
        ``a.reshape((10, 11))``.
        """
        order = 'C'
        if len(kwargs) > 1:
            raise TypeError('function takes at most 1 keyword argument')
        if len(kwargs) == 1:
            if 'order' not in kwargs:
                raise TypeError('{} is an invalid keyword argument for this function'
                                .format(kwargs.keys()[0]))
            order = kwargs.pop('order', 'C')
            if order != 'C':
                raise NotImplementedError('only supports C-order,'
                                          ' while received {}'.format(order))
        if len(args) == 0:
            raise TypeError('reshape() takes exactly 1 argument (0 given)')
        if len(args) == 1 and isinstance(args[0], tuple):
            return _mx_np_op.reshape(self, newshape=args[0], order=order)
        else:
            return _mx_np_op.reshape(self, newshape=args, order=order)

    def reshape_like(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`reshape_like`.

        The arguments are the same as for :py:func:`reshape_like`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute reshape_like')

    def reshape_view(self, *shape, **kwargs):
        """Returns a **view** of this array with a new shape without altering any data.
        Inheritated from NDArray.reshape.
        """
        return super(ndarray, self).reshape(*shape, **kwargs)

    def zeros_like(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`zeros_like`.

        The arguments are the same as for :py:func:`zeros_like`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute zeros_like')

    def ones_like(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`ones_like`.

        The arguments are the same as for :py:func:`ones_like`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute ones_like')

    def broadcast_axes(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`broadcast_axes`.

        The arguments are the same as for :py:func:`broadcast_axes`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute broadcast_like')

    def repeat(self, repeats, axis=None):  # pylint: disable=arguments-differ
        """Repeat elements of an array."""
        raise NotImplementedError

    def pad(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`pad`.

        The arguments are the same as for :py:func:`pad`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute pad')

    def swapaxes(self, axis1, axis2):  # pylint: disable=arguments-differ
        """Return a copy of the array with axis1 and axis2 interchanged.
        Refer to `mxnet.numpy.swapaxes` for full documentation.
        """
        raise NotImplementedError

    def split(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`split`.

        The arguments are the same as for :py:func:`split`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute split')

    def split_v2(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`split_v2`.

        The arguments are the same as for :py:func:`split_v2`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute split_v2')

    def slice(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`slice`.

        The arguments are the same as for :py:func:`slice`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute slice')

    def slice_axis(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`slice_axis`.

        The arguments are the same as for :py:func:`slice_axis`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute slice_axis')

    def slice_like(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`slice_like`.

        The arguments are the same as for :py:func:`slice_like`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute slice_like')

    def slice_assign_scalar(self, value, begin, end, step):
        """
        Assign the scalar to a cropped subset of this ndarray. Value will broadcast to the shape of the cropped shape
        and will be cast to the same dtype of the ndarray.

        Parameters
        ----------
        value: numeric value
            Value and this ndarray should be of the same data type.
            The shape of rhs should be the same as the cropped shape of this ndarray.
        begin: tuple of begin indices
        end: tuple of end indices
        step: tuple of step lenghths

        Returns
        -------
        This ndarray.

        Examples
        --------
        >>> x = np.ones((2, 2, 2))
        >>> y = x.slice_assign_scalar(0, (0, 0, None), (1, 1, None), (None, None, None))
        >>> y
        array([[[0., 0.],
                [1., 1.]],

               [[1., 1.],
                [1., 1.]]])
        >>> x
        array([[[0., 0.],
                [1., 1.]],

               [[1., 1.],
                [1., 1.]]])
        """
        return _npi.slice_assign_scalar(self, value, begin=begin, end=end, step=step, out=self)

    def slice_assign(self, rhs, begin, end, step):
        """
        Assign the rhs to a cropped subset of this ndarray in place.
        Returns the view of this ndarray.

        Parameters
        ----------
        rhs: ndarray.
            rhs and this NDArray should be of the same data type, and on the same device.
            The shape of rhs should be the same as the cropped shape of this ndarray.
        begin: tuple of begin indices
        end: tuple of end indices
        step: tuple of step lenghths

        Returns
        -------
        out : ndarray
            This ndarray.

        Examples
        --------
        >>> x = np.ones((2, 2, 2))
        >>> assigned = np.zeros((1, 1, 2))
        >>> y = x.slice_assign(assigned, (0, 0, None), (1, 1, None), (None, None, None))
        >>> y
        array([[[0., 0.],
                [1., 1.]],

               [[1., 1.],
                [1., 1.]]])
        >>> x
        array([[[0., 0.],
                [1., 1.]],

               [[1., 1.],
                [1., 1.]]])
        """
        return _npi.slice_assign(self, rhs, begin=begin, end=end, step=step, out=self)

    def take(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`take`.

        The arguments are the same as for :py:func:`take`, with
        this array as data.
        """
        raise NotImplementedError

    def one_hot(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`one_hot`.

        The arguments are the same as for :py:func:`one_hot`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute one_hot')

    def pick(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`pick`.

        The arguments are the same as for :py:func:`pick`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute pick')

    def sort(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sort`.

        The arguments are the same as for :py:func:`sort`, with
        this array as data.
        """
        raise NotImplementedError

    def topk(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`topk`.

        The arguments are the same as for :py:func:`topk`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute topk')

    def argsort(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`argsort`.

        The arguments are the same as for :py:func:`argsort`, with
        this array as data.
        """
        raise NotImplementedError

    def argmax_channel(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`argmax_channel`.

        The arguments are the same as for :py:func:`argmax_channel`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute argmax_channel')

    def argmin(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`argmin`.

        The arguments are the same as for :py:func:`argmin`, with
        this array as data.
        """
        raise NotImplementedError

    def clip(self, min=None, max=None, out=None):  # pylint: disable=arguments-differ
        """Return an array whose values are limited to [min, max].
        One of max or min must be given.
        """
        raise NotImplementedError

    def abs(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`abs`.

        The arguments are the same as for :py:func:`abs`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute abs')

    def sign(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sign`.

        The arguments are the same as for :py:func:`sign`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute sign')

    def flatten(self, order='C'):  # pylint: disable=arguments-differ
        """Return a copy of the array collapsed into one dimension."""
        return self.reshape(-1, order=order)

    def shape_array(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`shape_array`.

        The arguments are the same as for :py:func:`shape_array`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute shape_array')

    def size_array(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`size_array`.

        The arguments are the same as for :py:func:`size_array`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute size_array')

    def expand_dims(self, *args, **kwargs):  # pylint: disable=arguments-differ,unused-argument
        """Convenience fluent method for :py:func:`expand_dims`.

        The arguments are the same as for :py:func:`expand_dims`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute expand_dims')

    def tile(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`tile`.

        The arguments are the same as for :py:func:`tile`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute tile')

    def transpose(self, *axes):  # pylint: disable=arguments-differ
        """Permute the dimensions of an array."""
        return _mx_np_op.transpose(self, axes=axes if len(axes) != 0 else None)

    def flip(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`flip`.

        The arguments are the same as for :py:func:`flip`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute flip')

    def depth_to_space(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`depth_to_space`.

        The arguments are the same as for :py:func:`depth_to_space`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute depth_to_space')

    def space_to_depth(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`space_to_depth`.

        The arguments are the same as for :py:func:`space_to_depth`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute space_to_depth')

    def diag(self, k=0, **kwargs):
        """Convenience fluent method for :py:func:`diag`.

        The arguments are the same as for :py:func:`diag`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute diag')

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):  # pylint: disable=arguments-differ
        """Return the sum of the array elements over the given axis."""
        return _mx_np_op.sum(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def nansum(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`nansum`.

        The arguments are the same as for :py:func:`nansum`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute nansum')

    def prod(self, axis=None, dtype=None, out=None, keepdims=False):  # pylint: disable=arguments-differ
        """Return the product of the array elements over the given axis."""
        return _mx_np_op.prod(self, axis=axis, dtype=dtype, keepdims=keepdims, out=out)

    def nanprod(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`nanprod`.

        The arguments are the same as for :py:func:`nanprod`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute nanprod')

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):  # pylint: disable=arguments-differ
        """Returns the average of the array elements along given axis."""
        raise NotImplementedError

    # TODO(junwu): Use mxnet std op instead of onp.std
    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):  # pylint: disable=arguments-differ
        """Returns the standard deviation of the array elements along given axis."""
        ret_np = self.asnumpy().std(axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)
        return array(ret_np, dtype=ret_np.dtype, ctx=self.context)

    def cumsum(self, axis=None, dtype=None, out=None):
        """Return the cumulative sum of the elements along the given axis."""
        raise NotImplementedError

    def tolist(self):
        return self.asnumpy().tolist()

    def max(self, axis=None, out=None, keepdims=False):  # pylint: disable=arguments-differ
        """Return the maximum along a given axis."""
        raise NotImplementedError

    def min(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`min`.

        The arguments are the same as for :py:func:`min`, with
        this array as data.
        """
        raise NotImplementedError

    def norm(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`norm`.

        The arguments are the same as for :py:func:`norm`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute norm')

    def round(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`round`.

        The arguments are the same as for :py:func:`round`, with
        this array as data.
        """
        raise NotImplementedError

    def rint(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`rint`.

        The arguments are the same as for :py:func:`rint`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute rint')

    def fix(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`fix`.

        The arguments are the same as for :py:func:`fix`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute fix')

    def floor(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`floor`.

        The arguments are the same as for :py:func:`floor`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute floor')

    def ceil(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`ceil`.

        The arguments are the same as for :py:func:`ceil`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute ceil')

    def trunc(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`trunc`.

        The arguments are the same as for :py:func:`trunc`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute trunc')

    def sin(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sin`.

        The arguments are the same as for :py:func:`sin`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute sin')

    def cos(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`cos`.

        The arguments are the same as for :py:func:`cos`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute cos')

    def tan(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`tan`.

        The arguments are the same as for :py:func:`tan`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute tan')

    def arcsin(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arcsin`.

        The arguments are the same as for :py:func:`arcsin`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute arcsin')

    def arccos(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arccos`.

        The arguments are the same as for :py:func:`arccos`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute arccos')

    def arctan(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arctan`.

        The arguments are the same as for :py:func:`arctan`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute arctan')

    def degrees(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`degrees`.

        The arguments are the same as for :py:func:`degrees`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute degrees')

    def radians(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`radians`.

        The arguments are the same as for :py:func:`radians`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute radians')

    def sinh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sinh`.

        The arguments are the same as for :py:func:`sinh`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute sinh')

    def cosh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`cosh`.

        The arguments are the same as for :py:func:`cosh`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute cosh')

    def tanh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`tanh`.

        The arguments are the same as for :py:func:`tanh`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute tanh')

    def arcsinh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arcsinh`.

        The arguments are the same as for :py:func:`arcsinh`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute arcsinh')

    def arccosh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arccosh`.

        The arguments are the same as for :py:func:`arccosh`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute arccosh')

    def arctanh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arctanh`.

        The arguments are the same as for :py:func:`arctanh`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute arctanh')

    def exp(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`exp`.

        The arguments are the same as for :py:func:`exp`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute exp')

    def expm1(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`expm1`.

        The arguments are the same as for :py:func:`expm1`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute expm1')

    def log(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log`.

        The arguments are the same as for :py:func:`log`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute log')

    def log10(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log10`.

        The arguments are the same as for :py:func:`log10`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute log10')

    def log2(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log2`.

        The arguments are the same as for :py:func:`log2`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute log2')

    def log1p(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log1p`.

        The arguments are the same as for :py:func:`log1p`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute log1p')

    def sqrt(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sqrt`.

        The arguments are the same as for :py:func:`sqrt`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute sqrt')

    def rsqrt(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`rsqrt`.

        The arguments are the same as for :py:func:`rsqrt`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute rsqrt')

    def cbrt(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`cbrt`.

        The arguments are the same as for :py:func:`cbrt`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute cqrt')

    def rcbrt(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`rcbrt`.

        The arguments are the same as for :py:func:`rcbrt`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute rcqrt')

    def square(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`square`.

        The arguments are the same as for :py:func:`square`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute square')

    def reciprocal(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`reciprocal`.

        The arguments are the same as for :py:func:`reciprocal`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute reciprocal')

    def relu(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`relu`.

        The arguments are the same as for :py:func:`relu`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute relu')

    def sigmoid(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sigmoid`.

        The arguments are the same as for :py:func:`sigmoid`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute sigmoid')

    def softmax(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`softmax`.

        The arguments are the same as for :py:func:`softmax`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute softmax')

    def log_softmax(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log_softmax`.

        The arguments are the same as for :py:func:`log_softmax`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute log_softmax')

    def softmin(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`softmin`.

        The arguments are the same as for :py:func:`softmin`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute softmin')

    def squeeze(self, axis=None):  # pylint: disable=arguments-differ
        """Remove single-dimensional entries from the shape of a."""
        return _mx_np_op.squeeze(self, axis=axis)

    def broadcast_to(self, shape):
        return _mx_np_op.broadcast_to(self, shape)

    def broadcast_like(self, other):
        raise AttributeError('mxnet.numpy.ndarray object has no attribute broadcast_like')

    def _full(self, value):
        """
        Currently for internal use only. Implemented for __setitem__.
        Assign to self an array of self's same shape and type, filled with value.
        """
        return _mx_nd_np.full(self.shape, value, ctx=self.context, dtype=self.dtype, out=self)

    def _scatter_set_nd(self, value_nd, indices):
        """
        This is added as an ndarray class method in order to support polymorphism in NDArray and numpy.ndarray indexing
        """
        return _npi.scatter_set_nd(
            lhs=self, rhs=value_nd, indices=indices, shape=self.shape, out=self
        )

    @property
    def shape(self):
        return super(ndarray, self).shape

    @property
    def ndim(self):
        """Number of array dimensions."""
        return len(self.shape)

    @property
    def size(self):
        """Number of elements in the array."""
        return super(ndarray, self).size

    def tostype(self, stype):
        raise AttributeError('mxnet.numpy.ndarray object has no attribute tostype')


@set_module('mxnet.numpy')
def empty(shape, dtype=float, order='C', ctx=None):
    """Return a new array of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or tuple of int Shape of the empty array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        Desired output data-type for the array, e.g, `numpy.int8`. Default is
        `numpy.float32`. Note that this behavior is different from NumPy's `empty`
        function where `float64` is the default value, because `float32` is
        considered as the default data type in deep learning.
    order : {'C'}, optional, default: 'C'
        How to store multi-dimensional data in memory, currently only row-major
        (C-style) is supported.
    ctx : device context, optional
        Device context on which the memory is allocated. Default is
        `mxnet.context.current_context()`.

    Returns
    -------
    out : ndarray
        Array of uninitialized (arbitrary) data of the given shape, dtype, and order.
    """
    if order != 'C':
        raise NotImplementedError
    if ctx is None:
        ctx = current_context()
    if dtype is None:
        dtype = _np.float32
    if isinstance(shape, int):
        shape = (shape,)
    return ndarray(handle=_new_alloc_handle(shape, ctx, False, dtype))


@set_module('mxnet.numpy')
def array(object, dtype=None, ctx=None):
    """
    Create an array.

    Parameters
    ----------
    object : array_like or `numpy.ndarray` or `mxnet.numpy.ndarray`
        An array, any object exposing the array interface, an object whose
        __array__ method returns an array, or any (nested) sequence.
    dtype : data-type, optional
        The desired data-type for the array. Default is `float32`.
    ctx : device context, optional
        Device context on which the memory is allocated. Default is
        `mxnet.context.current_context()`.

    Returns
    -------
    out : ndarray
        An array object satisfying the specified requirements.
    """
    if ctx is None:
        ctx = current_context()
    if isinstance(object, ndarray):
        dtype = object.dtype if dtype is None else dtype
    else:
        dtype = mx_real_t if dtype is None else dtype
        if not isinstance(object, (ndarray, _np.ndarray)):
            try:
                object = _np.array(object, dtype=dtype)
            except Exception as e:
                raise TypeError('{}'.format(str(e)))
    ret = empty(object.shape, dtype=dtype, ctx=ctx)
    if len(object.shape) == 0:
        ret[()] = object
    else:
        ret[:] = object
    return ret


@set_module('mxnet.numpy')
def zeros(shape, dtype=_np.float32, order='C', ctx=None):
    """Return a new array of given shape and type, filled with zeros.
    This function currently only supports storing multi-dimensional data
    in row-major (C-style).

    Parameters
    ----------
    shape : int or tuple of int
        The shape of the empty array.
    dtype : str or numpy.dtype, optional
        An optional value type (default is `numpy.float32`). Note that this
        behavior is different from NumPy's `ones` function where `float64`
        is the default value, because `float32` is considered as the default
        data type in deep learning.
    order : {'C'}, optional, default: 'C'
        How to store multi-dimensional data in memory, currently only row-major
        (C-style) is supported.
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    out : ndarray
        Array of zeros with the given shape, dtype, and ctx.
    """
    return _mx_nd_np.zeros(shape, dtype, order, ctx)


@set_module('mxnet.numpy')
def ones(shape, dtype=_np.float32, order='C', ctx=None):
    """Return a new array of given shape and type, filled with zeros.
    This function currently only supports storing multi-dimensional data
    in row-major (C-style).

    Parameters
    ----------
    shape : int or tuple of int
        The shape of the empty array.
    dtype : str or numpy.dtype, optional
        An optional value type. Default is `numpy.float32`. Note that this
        behavior is different from NumPy's `ones` function where `float64`
        is the default value, because `float32` is considered as the default
        data type in deep learning.
    order : {'C'}, optional, default: 'C'
        How to store multi-dimensional data in memory, currently only row-major
        (C-style) is supported.
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    out : ndarray
        Array of zeros with the given shape, dtype, and ctx.
    """
    return _mx_nd_np.ones(shape, dtype, order, ctx)


@set_module('mxnet.numpy')
def full(shape, fill_value, dtype=None, order='C', ctx=None, out=None):  # pylint: disable=too-many-arguments
    """
    Return a new array of given shape and type, filled with `fill_value`.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    fill_value : scalar
        Fill value.
    dtype : data-type, optional
        The desired data-type for the array. The default, `None`, means
        `np.array(fill_value).dtype`.
    order : {'C'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory. Currently only supports C order.
    ctx: to specify the device, e.g. the i-th GPU.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.

    Returns
    -------
    out : ndarray
        Array of `fill_value` with the given shape, dtype, and order.

    Notes
    -----
    This function differs from the original `numpy.full
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.full.html`_ in
    the following way(s):

    - Has an additional `ctx` argument to specify the device
    - Has an additional `out` argument
    - Currently does not support `order` selection

    See Also
    --------
    empty : Return a new uninitialized array.
    ones : Return a new array setting values to one.
    zeros : Return a new array setting values to zero.

    Examples
    --------
    >>> np.full((2, 2), 10)
    array([[10., 10.],
           [10., 10.]])
    >>> np.full((2, 2), 2, dtype=np.int32, ctx=mx.cpu(0))
    array([[2, 2],
           [2, 2]], dtype=int32)
    """
    return _mx_nd_np.full(shape, fill_value, order=order, ctx=ctx, dtype=dtype, out=out)


@set_module('mxnet.numpy')
def add(x1, x2, out=None):
    """Add arguments element-wise.

    Parameters
    ----------
    x1, x2 : ndarrays or scalar values
        The arrays to be added. If x1.shape != x2.shape, they must be broadcastable to
        a common shape (which may be the shape of one or the other).

    out : ndarray
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned.

    Returns
    -------
    add : ndarray or scalar
        The sum of x1 and x2, element-wise. This is a scalar if both x1 and x2 are scalars.
    """
    return _mx_nd_np.add(x1, x2, out)


@set_module('mxnet.numpy')
def subtract(x1, x2, out=None):
    """Subtract arguments element-wise.

    Parameters
    ----------
    x1, x2 : ndarrays or scalar values
        The arrays to be subtracted from each other. If x1.shape != x2.shape,
        they must be broadcastable to a common shape (which may be the shape
        of one or the other).

    out : ndarray
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned.

    Returns
    -------
    subtract : ndarray or scalar
        The difference of x1 and x2, element-wise. This is a scalar if both x1 and x2 are scalars.
    """
    return _mx_nd_np.subtract(x1, x2, out)


@set_module('mxnet.numpy')
def multiply(x1, x2, out=None):
    """Multiply arguments element-wise.

    Parameters
    ----------
    x1, x2 : ndarrays or scalar values
        The arrays to be multiplied. If x1.shape != x2.shape, they must be broadcastable to
        a common shape (which may be the shape of one or the other).

    out : ndarray
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned.

    Returns
    -------
    out : ndarray or scalar
        The difference of x1 and x2, element-wise. This is a scalar if both x1 and x2 are scalars.
    """
    return _mx_nd_np.multiply(x1, x2, out)


@set_module('mxnet.numpy')
def divide(x1, x2, out=None):
    """Returns a true division of the inputs, element-wise.

    Parameters
    ----------
    x1 : ndarray or scalar
        Dividend array.

    x2 : ndarray or scalar
        Divisor array.

    out : ndarray
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned.

    Returns
    -------
    out : ndarray or scalar
        This is a scalar if both x1 and x2 are scalars.
    """
    return _mx_nd_np.divide(x1, x2, out=out)


@set_module('mxnet.numpy')
def mod(x1, x2, out=None):
    """Return element-wise remainder of division.

    Parameters
    ----------
    x1 : ndarray or scalar
        Dividend array.

    x2 : ndarray or scalar
        Divisor array.

    out : ndarray
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned.

    Returns
    -------
    out : ndarray or scalar
        This is a scalar if both x1 and x2 are scalars.
    """
    return _mx_nd_np.mod(x1, x2, out=out)


@set_module('mxnet.numpy')
def power(x1, x2, out=None):
    """First array elements raised to powers from second array, element-wise.

    Parameters
    ----------
    x1 : ndarray or scalar
        The bases.

    x2 : ndarray or scalar
        The exponent.

    out : ndarray
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned.

    Returns
    -------
    out : ndarray or scalar
        The bases in x1 raised to the exponents in x2.
        This is a scalar if both x1 and x2 are scalars.
    """
    return _mx_nd_np.power(x1, x2, out=out)


@set_module('mxnet.numpy')
def sin(x, out=None, **kwargs):
    r"""Trigonometric sine, element-wise.
    Parameters
    ----------
    x : ndarray or scalar
        Angle, in radians (:math:`2 \pi` rad equals 360 degrees).
    out : ndarray or None
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output is the same as that of the input if the input is an ndarray.
    Returns
    -------
    y : ndarray or scalar
        The sine of each element of x. This is a scalar if `x` is a scalar.
    Notes
    ----
    This function only supports input type of float.
    """
    return _mx_nd_np.sin(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def cos(x, out=None, **kwargs):
    r"""Cosine, element-wise.
    Parameters
    ----------
    x : ndarray or scalar
        Angle, in radians (:math:`2 \pi` rad equals 360 degrees).
    out : ndarray or None
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output is the same as that of the input if the input is an ndarray.
    Returns
    -------
    y : ndarray or scalar
        The corresponding cosine values. This is a scalar if x is a scalar.
    Notes
    ----
    This function only supports input type of float.
    """
    return _mx_nd_np.cos(x, out=out, **kwargs)


def sinh(x, out=None, **kwargs):
    """Hyperbolic sine, element-wise.
    Equivalent to ``1/2 * (np.exp(x) - np.exp(-x))`` or ``-1j * np.sin(1j*x)``.
    Parameters
    ----------
    x : ndarray or scalar
        Input array or scalar.
    out : ndarray or None
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output is the same as that of the input if the input is an ndarray.
    Returns
    -------
    y : ndarray or scalar
        The corresponding hyperbolic sine values. This is a scalar if `x` is a scalar.
    Notes
    ----
    This function only supports input type of float.
    """
    return _mx_nd_np.sinh(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def cosh(x, out=None, **kwargs):
    """Hyperbolic cosine, element-wise.
    Equivalent to ``1/2 * (np.exp(x) + np.exp(-x))`` and ``np.cos(1j*x)``.
    Parameters
    ----------
    x : ndarray or scalar
        Input array or scalar.
    out : ndarray or None
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output is the same as that of the input if the input is an ndarray.
    Returns
    -------
    y : ndarray or scalar
        The corresponding hyperbolic cosine values. This is a scalar if `x` is a scalar.
    Notes
    ----
    This function only supports input type of float.
    """
    return _mx_nd_np.cosh(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def tanh(x, out=None, **kwargs):
    """
    Compute hyperbolic tangent element-wise.
    Equivalent to ``np.sinh(x)/np.cosh(x)``.
    Parameters
    ----------
    x : ndarray or scalar.
        Input array.
    out : ndarray or None
        A location into which the result is stored. If provided, it
        must have a shape that the inputs fill into. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output and input must be the same.
    Returns
    ----------
    y : ndarray or scalar
       The corresponding hyperbolic tangent values.
    Notes
    -----
    If `out` is provided, the function writes the result into it,
    and returns a reference to `out`.  (See Examples)
    - input x does not support complex computation (like imaginary number)
    >>> np.tanh(np.pi*1j)
    TypeError: type <type 'complex'> not supported
    Examples
    --------
    >>> np.tanh(np.array[0, np.pi]))
    array([0.       , 0.9962721])
    >>> np.tanh(np.pi)
    0.99627207622075
    >>> # Example of providing the optional output parameter illustrating
    >>> # that what is returned is a reference to said parameter
    >>> out1 = np.array(1)
    >>> out2 = np.tanh(np.array(0.1), out1)
    >>> out2 is out1
    True
    >>> # Example of ValueError due to provision of shape mis-matched `out`
    >>> np.tanh(np.zeros((3,3)),np.zeros((2,2)))
    mxnet.base.MXNetError:
    [07:17:36] ../src/ndarray/./../operator/tensor/../elemwise_op_common.h:135:
    Check failed: assign(&dattr, vec.at(i)): Incompatible attr in node
    at 0-th output: expected [3,3], got [2,2]
    """
    return _mx_nd_np.tanh(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def log10(x, out=None, **kwargs):
    """Return the base 10 logarithm of the input array, element-wise.
    Parameters
    ----------
    x : ndarray or scalar
        Input array or scalar.
    out : ndarray or None
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output is the same as that of the input if the input is an ndarray.
    Returns
    -------
    y : ndarray or scalar
        The logarithm to the base 10 of `x`, element-wise. NaNs are
        returned where x is negative. This is a scalar if `x` is a scalar.
    Notes
    ----
    This function only supports input type of float.
    """
    return _mx_nd_np.log10(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def sqrt(x, out=None, **kwargs):
    """
    Return the non-negative square-root of an array, element-wise.
    Parameters
    ----------
    x : ndarray or scalar
        The values whose square-roots are required.
    out : ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    y : ndarray or scalar
        An array of the same shape as `x`, containing the positive
        square-root of each element in `x`. This is a scalar if `x` is a scalar.
    Notes
    ----
    This function only supports input type of float.
    """
    return _mx_nd_np.sqrt(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def cbrt(x, out=None, **kwargs):
    """
    Return the cube-root of an array, element-wise.
    Parameters
    ----------
    x : ndarray
        The values whose cube-roots are required.
    out : ndarray, optional
        A location into which the result is stored. If provided, it must have a shape that the
        inputs broadcast to. If not provided or None, a freshly-allocated array is returned.
        A tuple (possible only as a keyword argument) must have length equal to the number of outputs.
    Returns
    ----------
    y : ndarray
        An array of the same shape as x, containing the cube cube-root of each element in x.
        If out was provided, y is a reference to it. This is a scalar if x is a scalar.
    Examples
    ----------
    >>> np.cbrt([1,8,27])
    array([ 1.,  2.,  3.])
    """
    return _mx_nd_np.cbrt(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def abs(x, out=None, **kwargs):
    r"""abs(x, out=None, **kwargs)
    Calculate the absolute value element-wise.
    Parameters
    ----------
    x : ndarray or scalar
        Input array.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    absolute : ndarray
        An ndarray containing the absolute value of
        each element in `x`. This is a scalar if `x` is a scalar.
    Examples
    --------
    >>> x = np.array([-1.2, 1.2])
    >>> np.abs(x)
    array([1.2, 1.2])
    """
    return _mx_nd_np.abs(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def absolute(x, out=None, **kwargs):
    """
    Calculate the absolute value element-wise.
    np.abs is a shorthand for this function.
    Parameters
    ----------
    x : ndarray
        Input array.
    out : ndarray, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array is returned.
        A tuple (possible only as a keyword argument) must have length equal to the number of outputs.
    Returns
    ----------
    absolute : ndarray
        An ndarray containing the absolute value of each element in x.
    Examples
    ----------
    >>> x = np.array([-1.2, 1.2])
    >>> np.absolute(x)
    array([ 1.2,  1.2])
    """
    return _mx_nd_np.absolute(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def exp(x, out=None, **kwargs):
    r"""exp(x, out=None, **kwargs)
    Calculate the exponential of all elements in the input array.
    Parameters
    ----------
    x : ndarray or scalar
        Input values.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    out : ndarray or scalar
        Output array, element-wise exponential of `x`.
        This is a scalar if `x` is a scalar.
    Examples
    --------
    >>> np.exp(1)
    2.718281828459045
    >>> x = np.array([-1, 1, -2, 2])
    >>> np.exp(x)
    array([0.36787945, 2.7182817 , 0.13533528, 7.389056  ])
    """
    return _mx_nd_np.exp(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def expm1(x, out=None, **kwargs):
    r"""expm1(x, out=None, **kwargs)
    Calculate `exp(x) - 1` for all elements in the array.
    Parameters
    ----------
    x : ndarray or scalar
        Input values.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    out : ndarray or scalar
        Output array, element-wise exponential minus one: `out = exp(x) - 1`.
        This is a scalar if `x` is a scalar.
    Examples
    --------
    >>> np.expm1(1)
    1.718281828459045
    >>> x = np.array([-1, 1, -2, 2])
    >>> np.exp(x)
    array([-0.63212056,  1.71828183, -0.86466472,  6.3890561])
    """
    return _mx_nd_np.expm1(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def arcsin(x, out=None, **kwargs):
    r"""
    arcsin(x, out=None)
    Inverse sine, element-wise.
    Parameters
    ----------
    x : ndarray or scalar
        `y`-coordinate on the unit circle.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape as the input.
        If not provided or None, a freshly-allocated array is returned.
    Returns
    -------
    angle : ndarray or scalar
        Output array is same shape and type as x. This is a scalar if x is a scalar.
        The inverse sine of each element in `x`, in radians and in the
        closed interval ``[-pi/2, pi/2]``.
    Examples
    --------
    >>> np.arcsin(1)     # pi/2
    1.5707963267948966
    >>> np.arcsin(-1)    # -pi/2
    -1.5707963267948966
    >>> np.arcsin(0)
    0.0
    Notes
    -----
    `arcsin` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that :math:`sin(z) = x`.  The convention is to
    return the angle `z` whose real part lies in [-pi/2, pi/2].
    For real-valued input data types, *arcsin* always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.
    The inverse sine is also known as `asin` or sin^{-1}.
    The output `ndarray` has the same `ctx` as the input `ndarray`.
    This function differs from the original `numpy.arcsin
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.arcsin.html>`_ in
    the following aspects:
    - Only support ndarray or scalar now.
    - `where` argument is not supported.
    - Complex input is not supported.
    References
    ----------
    Abramowitz, M. and Stegun, I. A., *Handbook of Mathematical Functions*,
    10th printing, New York: Dover, 1964, pp. 79ff.
    http://www.math.sfu.ca/~cbm/aands/
    """
    return _mx_nd_np.arcsin(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def arccos(x, out=None, **kwargs):
    """
    Trigonometric inverse cosine, element-wise.
    The inverse of cos so that, if y = cos(x), then x = arccos(y).
    Parameters
    ----------
    x : ndarray
        x-coordinate on the unit circle. For real arguments, the domain is [-1, 1].
    out : ndarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to. If not provided or None, a freshly-allocated array is returned.
        A tuple (possible only as a keyword argument) must have length equal to the number of outputs.
    Returns
    ----------
    angle : ndarray
        The angle of the ray intersecting the unit circle at the given x-coordinate in radians [0, pi].
        This is a scalar if x is a scalar.
    See also
    ----------
    cos, arctan, arcsin
    Notes
    ----------
    arccos is a multivalued function: for each x there are infinitely many numbers z such that
    cos(z) = x. The convention is to return the angle z whose real part lies in [0, pi].
    For real-valued input data types, arccos always returns real output.
    For each value that cannot be expressed as a real number or infinity, it yields nan and sets
    the invalid floating point error flag.
    The inverse cos is also known as acos or cos^-1.
    Examples
    ----------
    We expect the arccos of 1 to be 0, and of -1 to be pi:
    >>> np.arccos([1, -1])
    array([ 0.        ,  3.14159265])
    """
    return _mx_nd_np.arccos(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def arctan(x, out=None, **kwargs):
    r"""arctan(x, out=None, **kwargs)
    Trigonometric inverse tangent, element-wise.
    The inverse of tan, so that if ``y = tan(x)`` then ``x = arctan(y)``.
    Parameters
    ----------
    x : ndarray or scalar
        Input values.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    out : ndarray or scalar
        Out has the same shape as `x`. It lies is in
        ``[-pi/2, pi/2]`` (``arctan(+/-inf)`` returns ``+/-pi/2``).
        This is a scalar if `x` is a scalar.
    Notes
    -----
    `arctan` is a multi-valued function: for each `x` there are infinitely
    many numbers `z` such that tan(`z`) = `x`.  The convention is to return
    the angle `z` whose real part lies in [-pi/2, pi/2].
    For real-valued input data types, `arctan` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.
    For complex-valued input, we do not have support for them yet.
    The inverse tangent is also known as `atan` or tan^{-1}.
    Examples
    --------
    We expect the arctan of 0 to be 0, and of 1 to be pi/4:
    >>> x = np.array([0, 1])
    >>> np.arctan(x)
    array([0.       , 0.7853982])
    >>> np.pi/4
    0.7853981633974483
    """
    return _mx_nd_np.arctan(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def sign(x, out=None):
    """
    sign(x, out=None)
    Returns an element-wise indication of the sign of a number.
    The `sign` function returns ``-1 if x < 0, 0 if x==0, 1 if x > 0``. Only supports real number.
    Parameters
    ----------
    x : ndarray or a scalar
        Input values.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.
    Returns
    -------
    y : ndarray
        The sign of `x`.
        This is a scalar if `x` is a scalar.
    Note
    -------
    - Only supports real number as input elements.
    - Input type does not support Python native iterables(list, tuple, ...).
    - ``out`` param: cannot perform auto broadcasting. ``out`` ndarray's shape must be the same as the expected output.
    - ``out`` param: cannot perform auto type cast. ``out`` ndarray's dtype must be the same as the expected output.
    - ``out`` param does not support scalar input case.
    Examples
    --------
    >>> a = np.array([-5., 4.5])
    >>> np.sign(a)
    array([-1.,  1.])
    Scalars as input:
    >>> np.sign(4.0)
    1.0
    >>> np.sign(0)
    0
    Use ``out`` parameter:
    >>> b = np.zeros((2, ))
    >>> np.sign(a, out=b)
    array([-1.,  1.])
    >>> b
    array([-1.,  1.])
    """
    return _mx_nd_np.sign(x, out=out)


@set_module('mxnet.numpy')
def log(x, out=None, **kwargs):
    """
    log(x, out=None)
    Natural logarithm, element-wise.
    The natural logarithm `log` is the inverse of the exponential function,
    so that `log(exp(x)) = x`. The natural logarithm is logarithm in base
    `e`.
    Parameters
    ----------
    x : ndarray
        Input value. Elements must be of real value.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.
    Returns
    -------
    y : ndarray
        The natural logarithm of `x`, element-wise.
        This is a scalar if `x` is a scalar.
    Notes
    -----
    Currently only supports data of real values and ``inf`` as input. Returns data of real value, ``inf``, ``-inf`` and
    ``nan`` according to the input.
    This function differs from the original `numpy.log
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html>`_ in
    the following aspects:
    - Does not support complex number for now
    - Input type does not support Python native iterables(list, tuple, ...).
    - ``out`` param: cannot perform auto broadcasting. ``out`` ndarray's shape must be the same as the expected output.
    - ``out`` param: cannot perform auto type cast. ``out`` ndarray's dtype must be the same as the expected output.
    - ``out`` param does not support scalar input case.
    Examples
    --------
    >>> a = np.array([1, np.exp(1), np.exp(2), 0], dtype=np.float64)
    >>> np.log(a)
    array([  0.,   1.,   2., -inf], dtype=float64)
    Due to internal calculation mechanism, using default float32 dtype may cause some special behavior:
    >>> a = np.array([1, np.exp(1), np.exp(2), 0])
    >>> np.log(a)
    array([  0.,  0.99999994,   2., -inf])
    Scalar calculation:
    >>> np.log(1)
    0.0
    """
    return _mx_nd_np.log(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def rint(x, out=None, **kwargs):
    """
    Round elements of the array to the nearest integer.
    Parameters
    ----------
    x : ndarray or scalar
        Input array.
    out : ndarray or None
        A location into which the result is stored.
        If provided, it must have the same shape and type as the input.
        If not provided or None, a freshly-allocated array is returned.
    Returns
    -------
    out : ndarray or scalar
        Output array is same shape and type as x. This is a scalar if x is a scalar.
    Notes
    -----
    This function differs from the original `numpy.rint
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.rint.html>`_ in
    the following way(s):
    - only ndarray or scalar is accpted as valid input, tuple of ndarray is not supported
    - broadcasting to `out` of different shape is currently not supported
    - when input is plain python numerics, the result will not be stored in the `out` param
    Examples
    --------
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> np.rint(a)
    array([-2., -2., -0.,  0.,  1.,  2.,  2.])
    """
    return _mx_nd_np.rint(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def log2(x, out=None, **kwargs):
    """
    Base-2 logarithm of x.
    Parameters
    ----------
    x : ndarray or scalar
        Input values.
    out : ndarray or None
        A location into which the result is stored.
        If provided, it must have the same shape and type as the input.
        If not provided or None, a freshly-allocated array is returned.
    Returns
    -------
    y : ndarray
        The logarithm base two of `x`, element-wise.
        This is a scalar if `x` is a scalar.
    Notes
    -----
    This function differs from the original `numpy.log2
    <https://www.google.com/search?q=numpy+log2>`_ in
    the following way(s):
    - only ndarray or scalar is accpted as valid input, tuple of ndarray is not supported
    - broadcasting to `out` of different shape is currently not supported
    - when input is plain python numerics, the result will not be stored in the `out` param
    Examples
    --------
    >>> x = np.array([0, 1, 2, 2**4])
    >>> np.log2(x)
    array([-inf,   0.,   1.,   4.])
    """
    return _mx_nd_np.log2(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def log1p(x, out=None, **kwargs):
    """
    Return the natural logarithm of one plus the input array, element-wise.
    Calculates ``log(1 + x)``.
    Parameters
    ----------
    x : ndarray or scalar
        Input array.
    out : ndarray or None
        A location into which the result is stored. If provided, it
        must have a shape that the inputs fill into. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output and input must be the same.
    Returns
    -------
    y : ndarray or scalar
        Natural logarithm of 1 + x, element-wise. This is a scalar
        if x is a scalar.
    Notes
    -----
    For real-valued input, `log1p` is accurate also for `x` so small
    that `1 + x == 1` in floating-point accuracy.
    Logarithm is a multivalued function: for each `x` there is an infinite
    number of `z` such that `exp(z) = 1 + x`. The convention is to return
    the `z` whose imaginary part lies in `[-pi, pi]`.
    For real-valued input data types, `log1p` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.
    cannot support complex-valued input.
    Examples
    --------
    >>> np.log1p(1e-99)
    1e-99
    >>> a = np.array([3, 4, 5])
    >>> np.log1p(a)
    array([1.3862944, 1.609438 , 1.7917595])
    """
    return _mx_nd_np.log1p(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def degrees(x, out=None, **kwargs):
    """
    degrees(x, out=None)
    Convert angles from radians to degrees.
    Parameters
    ----------
    x : ndarray
        Input value. Elements must be of real value.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.
    Returns
    -------
    y : ndarray
        The corresponding degree values; if `out` was supplied this is a
        reference to it.
        This is a scalar if `x` is a scalar.
    Notes
    -------
    This function differs from the original `numpy.degrees
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.degrees.html>`_ in
    the following aspects:
    - Input type does not support Python native iterables(list, tuple, ...). Only ndarray is supported.
    - ``out`` param: cannot perform auto broadcasting. ``out`` ndarray's shape must be the same as the expected output.
    - ``out`` param: cannot perform auto type cast. ``out`` ndarray's dtype must be the same as the expected output.
    - ``out`` param does not support scalar input case.
    Examples
    --------
    Convert a radian array to degrees
    >>> rad = np.arange(12.) * np.pi / 6
    >>> np.degrees(rad)
    array([  0.,  30.,  60.,  90., 120., 150., 180., 210., 240., 270., 300., 330.])
    Use specified ``out`` ndarray:
    >>> out = np.zeros((rad.shape))
    >>> np.degrees(rad, out)
    array([  0.,  30.,  60.,  90., 120., 150., 180., 210., 240., 270., 300., 330.])
    >>> out
    array([  0.,  30.,  60.,  90., 120., 150., 180., 210., 240., 270., 300., 330.])
    """
    return _mx_nd_np.degrees(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def radians(x, out=None, **kwargs):
    """
    Convert angles from degrees to radians.
    Parameters
    ----------
    x : ndarray or scalar
        Input array in degrees.
    out : ndarray or None
        A location into which the result is stored.
        If provided, it must have the same shape and type as the input.
        If not provided or None, a freshly-allocated array is returned.
    Returns
    -------
    y : ndarray
        The corresponding radian values. This is a scalar if x is a scalar.
    Notes
    -----
    This function differs from the original `numpy.radians
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.radians.html>`_ in
    the following way(s):
    - only ndarray or scalar is accpted as valid input, tuple of ndarray is not supported
    - broadcasting to `out` of different shape is currently not supported
    - when input is plain python numerics, the result will not be stored in the `out` param
    Examples
    --------
    >>> deg = np.arange(12.) * 30.
    >>> np.radians(deg)
    array([0.       , 0.5235988, 1.0471976, 1.5707964, 2.0943952, 2.6179938,
           3.1415927, 3.6651914, 4.1887903, 4.712389 , 5.2359877, 5.7595863],
           dtype=float32)
    """
    return _mx_nd_np.radians(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def reciprocal(x, out=None, **kwargs):
    r"""
    reciprocal(x, out=None)
    Return the reciprocal of the argument, element-wise.
    Calculates ``1/x``.
    Parameters
    ----------
    x : ndarray or scalar
        The values whose reciprocals are required.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape as the input.
        If not provided or None, a freshly-allocated array is returned.
    Returns
    -------
    y : ndarray or scalar
        Output array is same shape and type as x. This is a scalar if x is a scalar.
    Examples
    --------
    >>> np.reciprocal(2.)
    0.5
    >>> x = np.array([1, 2., 3.33])
    >>> np.reciprocal(x)
    array([1.       , 0.5      , 0.3003003])
    Notes
    -----
    .. note::
        This function is not designed to work with integers.
    For integer arguments with absolute value larger than 1 the result is
    always zero because of the way Python handles integer division.  For
    integer zero the result is an overflow.
    The output `ndarray` has the same `ctx` as the input `ndarray`.
    This function differs from the original `numpy.reciprocal
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.reciprocal.html>`_ in
    the following aspects:
    - Only support ndarray and scalar now.
    - `where` argument is not supported.
    """
    return _mx_nd_np.reciprocal(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def square(x, out=None, **kwargs):
    r"""
    square(x, out=None)
    Return the element-wise square of the input.
    Parameters
    ----------
    x : ndarray or scalar
        The values whose squares are required.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape as the input.
        If not provided or None, a freshly-allocated array is returned.
    Returns
    -------
    y : ndarray or scalar
        Output array is same shape and type as x. This is a scalar if x is a scalar.
    Examples
    --------
    >>> np.square(2.)
    4.0
    >>> x = np.array([1, 2., -1])
    >>> np.square(x)
    array([1., 4., 1.])
    Notes
    -----
    The output `ndarray` has the same `ctx` as the input `ndarray`.
    This function differs from the original `numpy.square
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.square.html>`_ in
    the following aspects:
    - Only support ndarray and scalar now.
    - `where` argument is not supported.
    - Complex input is not supported.
    """
    return _mx_nd_np.square(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def negative(x, out=None, where=True, **kwargs):
    r"""
    negative(x, out=None, where=True)
    Numerical negative, element-wise.
    Parameters:
    ------------
    x : ndarray or scalar
        Input array.
    out : ndarray, None, or tuple of ndarray and None, optional
          A location into which the result is stored.
          If provided, it must have a shape that the inputs broadcast to.
          If not provided or None, a freshly-allocated array is returned.
          A tuple (possible only as a keyword argument) must have length
          equal to the number of outputs.
    where : ndarray, optional
            Values of True indicate to calculate the ufunc at that position,
            values of False indicate to leave the value in the output alone.
    Returns:
    -------
    y : ndarray or scalar
        Returned array or scalar: y = -x. This is a scalar if x is a scalar.
    Examples:
    --------
    >>> np.negative(1)
    -1
    """
    return _mx_nd_np.negative(x, out=out)


@set_module('mxnet.numpy')
def fix(x, out=None):
    """
    Round an array of floats element-wise to nearest integer towards zero.
    The rounded values are returned as floats.

    Parameters:
    ----------
    x : ndarray
        An array of floats to be rounded
    out : ndarray, optional
        Output array
    Returns:
    -------
    y : ndarray or scalar
    Returned array or scalar: y = -x. This is a scalar if x is a scalar.ndarray of floats
    Examples:
    ---------
    >>> np.fix(3.14)
    3
    """
    return _mx_nd_np.fix(x, out=out)


@set_module('mxnet.numpy')
def tan(x, out=None, where=True, **kwargs):
    r"""
    tan(x, out=None, where=True)
    Compute tangent element-wise.
    Equivalent to np.sin(x)/np.cos(x) element-wise.

    Parameters:
    ----------
    x : ndarray
        Input array.
    out : ndarray or none, optional
          A location into which the result is stored. If provided,
          it must have a shape that the inputs broadcast to. If not provided or None,
          a freshly-allocated array is returned. A tuple (possible only as a keyword argument)
          must have length equal to the number of outputs.
    where : ndarray, optional
            Values of True indicate to calculate the ufunc at that position,
            values of False indicate to leave the value in the output alone.
    Returns:
    -------
    y : ndarray
    The corresponding tangent values. This is a scalar if x is a scalar.
    Examples:
    ---------
    >>> np.tan(0.5)
    0.5463024898437905
    """

    return _mx_nd_np.tan(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def ceil(x, out=None, **kwargs):
    r"""
    Return the ceiling of the input, element-wise.
    The ceil of the ndarray `x` is the smallest integer `i`, such that
    `i >= x`.  It is often denoted as :math:`\lceil x \rceil`.
    Parameters
    ----------
    x : ndarray or scalar
        Input array.
    out : ndarray or None
        A location into which the result is stored. If provided, it
        must have a shape that the inputs fill into. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output and input must be the same.
    Returns
    -------
    y : ndarray or scalar
        The ceiling of each element in `x`, with `float` dtype.
        This is a scalar if `x` is a scalar.
    Examples
    --------
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> np.ceil(a)
    array([-1., -1., -0.,  1.,  2.,  2.,  2.])
    >>> #if you use parameter out, x and out must be ndarray. if not, you will get an error!
    >>> a = np.array(1)
    >>> np.ceil(np.array(3.5), a)
    array(4.)
    >>> a
    array(4.)
    """
    return _mx_nd_np.ceil(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def floor(x, out=None, **kwargs):
    r"""
    Return the floor of the input, element-wise.
    The ceil of the ndarray `x` is the largest integer `i`, such that
    `i <= x`.  It is often denoted as :math:`\lfloor x \rfloor`.
    Parameters
    ----------
    x : ndarray or scalar
        Input array.
    out : ndarray or None
        A location into which the result is stored. If provided, it
        must have a shape that the inputs fill into. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output and input must be the same.
    Returns
    -------
    y : ndarray or scalar
        The floor of each element in `x`, with `float` dtype.
        This is a scalar if `x` is a scalar.
    Examples
    --------
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> np.floor(a)
    array([-2., -2., -1.,  0.,  1.,  1.,  2.])
    >>> #if you use parameter out, x and out must be ndarray. if not, you will get an error!
    >>> a = np.array(1)
    >>> np.floor(np.array(3.5), a)
    array(3.)
    >>> a
    array(3.)
    """
    return _mx_nd_np.floor(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def trunc(x, out=None, **kwargs):
    r"""
    trunc(x, out=None)
    Return the truncated value of the input, element-wise.
    The truncated value of the scalar `x` is the nearest integer `i` which
    is closer to zero than `x` is. In short, the fractional part of the
    signed number `x` is discarded.

    Parameters
    ----------
    x : ndarray or scalar
        Input data.
    out : ndarray or None, optional
        A location into which the result is stored.

    Returns
    -------
    y : ndarray or scalar
        The truncated value of each element in `x`.
        This is a scalar if `x` is a scalar.
    Notes
    -----
    This function differs from the original numpy.trunc in the following aspects:
        - Do not support `where`, a parameter in numpy which indicates where to calculate.
        - Cannot cast type automatically. Dtype of `out` must be same as the expected one.
        - Cannot broadcast automatically. Shape of `out` must be same as the expected one.
        - If `x` is plain python numeric, the result won't be stored in out.

    Examples
    --------
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> np.trunc(a)
    array([-1., -1., -0.,  0.,  1.,  1.,  2.])
    """
    return _mx_nd_np.trunc(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def logical_not(x, out=None, **kwargs):
    r"""
    logical_not(x, out=None)
    Compute the truth value of NOT x element-wise.
    Parameters
    ----------
    x : ndarray or scalar
        Logical NOT is applied to the elements of `x`.
    out : ndarray or None, optional
        A location into which the result is stored.

    Returns
    -------
    y : bool or ndarray of bool
        Boolean result with the same shape as `x` of the NOT operation
        on elements of `x`.
        This is a scalar if `x` is a scalar.
    Notes
    -----
    This function differs from the original numpy.logical_not in the following aspects:
        - Do not support `where`, a parameter in numpy which indicates where to calculate.
        - Cannot cast type automatically. Dtype of `out` must be same as the expected one.
        - Cannot broadcast automatically. Shape of `out` must be same as the expected one.
        - If `x` is plain python numeric, the result won't be stored in out.

    Examples
    --------
    >>> x= np.array([True, False, 0, 1])
    >>> np.logical_not(x)
    array([0., 1., 1., 0.])

    >>> x = np.arange(5)
    >>> np.logical_not(x<3)
    array([0., 0., 0., 1., 1.])
    """
    return _mx_nd_np.logical_not(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def arcsinh(x, out=None, **kwargs):
    r"""
    arcsinh(x, out=None)
    Inverse hyperbolic cosine, element-wise.
    Parameters
    ----------
    x : ndarray or scalar
        Input array.
    out : ndarray or None, optional
        A location into which the result is stored.

    Returns
    -------
    arcsinh : ndarray
        Array of the same shape as `x`.
        This is a scalar if `x` is a scalar.

    Notes
    -----
    `arcsinh` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `sinh(z) = x`.

    For real-valued input data types, `arcsinh` always returns real output.
    For each value that cannot be expressed as a real number or infinity, it
    yields ``nan`` and sets the `invalid` floating point error flag.

    This function differs from the original numpy.arcsinh in the following aspects:
        - Do not support `where`, a parameter in numpy which indicates where to calculate.
        - Do not support complex-valued input.
        - Cannot cast type automatically. DType of `out` must be same as the expected one.
        - Cannot broadcast automatically. Shape of `out` must be same as the expected one.
        - If `x` is plain python numeric, the result won't be stored in out.

    Examples
    --------
    >>> a = np.array([3.2, 5.0])
    >>> np.arcsinh(a)
    array([1.8309381, 2.2924316])

    >>> np.arcsinh(1)
    0.0
    """
    return _mx_nd_np.arcsinh(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def arccosh(x, out=None, **kwargs):
    r"""
    arccosh(x, out=None)
    Inverse hyperbolic cosine, element-wise.
    Parameters
    ----------
    x : ndarray or scalar
        Input array.
    out : ndarray or None, optional
        A location into which the result is stored.

    Returns
    -------
    arccosh : ndarray
        Array of the same shape as `x`.
        This is a scalar if `x` is a scalar.

    Notes
    -----
    `arccosh` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `cosh(z) = x`.

    For real-valued input data types, `arccosh` always returns real output.
    For each value that cannot be expressed as a real number or infinity, it
    yields ``nan`` and sets the `invalid` floating point error flag.

    This function differs from the original numpy.arccosh in the following aspects:
        - Do not support `where`, a parameter in numpy which indicates where to calculate.
        - Do not support complex-valued input.
        - Cannot cast type automatically. Dtype of `out` must be same as the expected one.
        - Cannot broadcast automatically. Shape of `out` must be same as the expected one.
        - If `x` is plain python numeric, the result won't be stored in out.

    Examples
    --------
    >>> a = np.array([3.2, 5.0])
    >>> np.arccosh(a)
    array([1.8309381, 2.2924316])

    >>> np.arccosh(1)
    0.0
    """
    return _mx_nd_np.arccosh(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def arctanh(x, out=None, **kwargs):
    r"""
    arctanh(x, out=None)
    Inverse hyperbolic tangent, element-wise.
    Parameters
    ----------
    x : ndarray or scalar
        Input array.
    out : ndarray or None, optional
        A location into which the result is stored.

    Returns
    -------
    arctanh : ndarray
        Array of the same shape as `x`.
        This is a scalar if `x` is a scalar.

    Notes
    -----
    `arctanh` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `tanh(z) = x`.

    For real-valued input data types, `arctanh` always returns real output.
    For each value that cannot be expressed as a real number or infinity, it
    yields ``nan`` and sets the `invalid` floating point error flag.

    This function differs from the original numpy.arctanh in the following aspects:
        - Do not support `where`, a parameter in numpy which indicates where to calculate.
        - Do not support complex-valued input.
        - Cannot cast type automatically. Dtype of `out` must be same as the expected one.
        - Cannot broadcast automatically. Shape of `out` must be same as the expected one.
        - If `x` is plain python numeric, the result won't be stored in out.

    Examples
    --------
    >>> a = np.array([0.0, -0.5])
    >>> np.arctanh(a)
    array([0., -0.54930615])

    >>> np.arctanh(1)
    0.0
    """
    return _mx_nd_np.arctanh(x, out=out, **kwargs)


@set_module('mxnet.numpy')
def tensordot(a, b, axes=2):
    r"""
    tensordot(a, b, axes=2)
    Compute tensor dot product along specified axes for arrays >= 1-D.
    Given two tensors (arrays of dimension greater than or equal to one),
    `a` and `b`, and an ndarray object containing two ndarray
    objects, ``(a_axes, b_axes)``, sum the products of `a`'s and `b`'s
    elements (components) over the axes specified by ``a_axes`` and
    ``b_axes``. The third argument can be a single non-negative
    integer_like scalar, ``N``; if it is such, then the last ``N``
    dimensions of `a` and the first ``N`` dimensions of `b` are summed
    over.
    Parameters
    ----------
    a, b : ndarray, len(shape) >= 1
        Tensors to "dot".
    axes : int or (2,) ndarray
        * integer_like
        If an int N, sum over the last N axes of `a` and the first N axes
        of `b` in order. The sizes of the corresponding axes must match.
        * (2,) ndarray
        Or, a list of axes to be summed over, first sequence applying to `a`,
        second to `b`. Both elements ndarray must be of the same length.
    See Also
    --------
    dot, einsum
    Notes
    -----
    Three common use cases are:
        * ``axes = 0`` : tensor product :math:`a\otimes b`
        * ``axes = 1`` : tensor dot product :math:`a\cdot b`
        * ``axes = 2`` : (default) tensor double contraction :math:`a:b`
    When `axes` is integer_like, the sequence for evaluation will be: first
    the -Nth axis in `a` and 0th axis in `b`, and the -1th axis in `a` and
    Nth axis in `b` last.
    When there is more than one axis to sum over - and they are not the last
    (first) axes of `a` (`b`) - the argument `axes` should consist of
    two sequences of the same length, with the first axis to sum over given
    first in both sequences, the second axis second, and so forth.
    Examples
    --------
    >>> a = np.arange(60.).reshape(3,4,5)
    >>> b = np.arange(24.).reshape(4,3,2)
    >>> c = np.tensordot(a,b, axes=([1,0],[0,1]))
    >>> c.shape
    (5, 2)
    >>> c
    array([[ 4400.,  4730.],
           [ 4532.,  4874.],
           [ 4664.,  5018.],
           [ 4796.,  5162.],
           [ 4928.,  5306.]])
    """
    return _mx_nd_np.tensordot(a, b, axes)


@set_module('mxnet.numpy')
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0, ctx=None):  # pylint: disable=too-many-arguments
    r"""
    Return evenly spaced numbers over a specified interval.

    Returns num evenly spaced samples, calculated over the interval [start, stop].
    The endpoint of the interval can optionally be excluded.

    Parameters
    ----------
    start : real number
        The starting value of the sequence.
    stop : real number
        The end value of the sequence, unless endpoint is set to False. In
        that case, the sequence consists of all but the last of num + 1
        evenly spaced samples, so that stop is excluded. Note that the step
        size changes when endpoint is False.
    num : int, optional
        Number of samples to generate. Default is 50. Must be non-negative.
    endpoint : bool, optional
        If True, stop is the last sample. Otherwise, it is not included.
        Default is True.
    retstep : bool, optional
        If True, return (samples, step), where step is the spacing between samples.
    dtype : dtype, optional
        The type of the output array. If dtype is not given, infer the data
        type from the other input arguments.
    axis : int, optional
        The axis in the result to store the samples. Relevant only if start or
        stop are array-like. By default (0), the samples will be along a new
        axis inserted at the beginning. Use -1 to get an axis at the end.

    Returns
    -------
    samples : ndarray
        There are num equally spaced samples in the closed interval
        `[start, stop]` or the half-open interval `[start, stop)`
        (depending on whether endpoint is True or False).
    step : float, optional
        Only returned if retstep is True
        Size of spacing between samples.


    See Also
    --------
    arange : Similar to `linspace`, but uses a step size (instead of the
             number of samples).

    Examples
    --------
    >>> np.linspace(2.0, 3.0, num=5)
    array([2.  , 2.25, 2.5 , 2.75, 3.  ])
    >>> np.linspace(2.0, 3.0, num=5, endpoint=False)
    array([2. , 2.2, 2.4, 2.6, 2.8])
    >>> np.linspace(2.0, 3.0, num=5, retstep=True)
    (array([2.  , 2.25, 2.5 , 2.75, 3.  ]), 0.25)

    Graphical illustration:

    >>> import matplotlib.pyplot as plt
    >>> N = 8
    >>> y = np.zeros(N)
    >>> x1 = np.linspace(0, 10, N, endpoint=True)
    >>> x2 = np.linspace(0, 10, N, endpoint=False)
    >>> plt.plot(x1.asnumpy(), y.asnumpy(), 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.plot(x2.asnumpy(), (y + 0.5).asnumpy(), 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.ylim([-0.5, 1])
    (-0.5, 1)
    >>> plt.show()

    Notes
    -----

    This function differs from the original `numpy.linspace
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html>`_ in
    the following aspects:

    - `start` and `stop` do not support list, numpy ndarray and mxnet ndarray
    - axis could only be 0
    - There could be an additional `ctx` argument to specify the device, e.g. the i-th
      GPU.
    """
    return _mx_nd_np.linspace(start, stop, num, endpoint, retstep, dtype, axis, ctx)


@set_module('mxnet.numpy')
def expand_dims(a, axis):
    """Expand the shape of an array.

    Insert a new axis that will appear at the `axis` position in the expanded array shape.

    Parameters
    ----------
    a : ndarray
        Input array.
    axis : int
        Position in the expanded axes where the new axis is placed.

    Returns
    -------
    res : ndarray
        Output array. The number of dimensions is one greater than that of
        the input array.
    """
    return _npi.expand_dims(a, axis)


@set_module('mxnet.numpy')
def tile(A, reps):
    r"""
    Construct an array by repeating A the number of times given by reps.

    If `reps` has length ``d``, the result will have dimension of
    ``max(d, A.ndim)``.

    If ``A.ndim < d``, `A` is promoted to be d-dimensional by prepending new
    axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication,
    or shape (1, 1, 3) for 3-D replication. If this is not the desired
    behavior, promote `A` to d-dimensions manually before calling this
    function.

    If ``A.ndim > d``, `reps` is promoted to `A`.ndim by pre-pending 1's to it.
    Thus for an `A` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as
    (1, 1, 2, 2).

    Parameters
    ----------
    A : ndarray or scalar
        An input array or a scalar to repeat.
    reps : a single integer or tuple of integers
        The number of repetitions of `A` along each axis.

    Returns
    -------
    c : ndarray
        The tiled output array.

    Examples
    --------
    >>> a = np.array([0, 1, 2])
    >>> np.tile(a, 2)
    array([0., 1., 2., 0., 1., 2.])
    >>> np.tile(a, (2, 2))
    array([[0., 1., 2., 0., 1., 2.],
           [0., 1., 2., 0., 1., 2.]])
    >>> np.tile(a, (2, 1, 2))
    array([[[0., 1., 2., 0., 1., 2.]],
           [[0., 1., 2., 0., 1., 2.]]])

    >>> b = np.array([[1, 2], [3, 4]])
    >>> np.tile(b, 2)
    array([[1., 2., 1., 2.],
           [3., 4., 3., 4.]])
    >>> np.(b, (2, 1))
    array([[1., 2.],
           [3., 4.],
           [1., 2.],
           [3., 4.]])

    >>> c = np.array([1,2,3,4])
    >>> np.tile(c,(4,1))
    array([[1., 2., 3., 4.],
           [1., 2., 3., 4.],
           [1., 2., 3., 4.],
           [1., 2., 3., 4.]])

    Scalar as input:

    >>> np.tile(2, 3)
    array([2, 2, 2]) # repeating integer `2`

    """
    return _mx_nd_np.tile(A, reps)


@set_module('mxnet.numpy')
def arange(start, stop=None, step=1, dtype=None, ctx=None):
    """Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including `start` but excluding `stop`).
    For integer arguments the function is equivalent to the Python built-in
    `range` function, but returns an ndarray rather than a list.

    Parameters
    ----------
    start : number, optional
        Start of interval. The interval includes this value.  The default
        start value is 0.
    stop : number
        End of interval. The interval does not include this value, except
        in some cases where `step` is not an integer and floating point
        round-off affects the length of `out`.
    step : number, optional
        Spacing between values. For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified as a position argument,
        `start` must also be given.
    dtype : dtype
        The type of the output array. The default is `float32`.

    Returns
    -------
    arange : ndarray
        Array of evenly spaced values.

        For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``.  Because of floating point overflow,
        this rule may result in the last element of `out` being greater
        than `stop`.
    """
    return _mx_nd_np.arange(start, stop, step, dtype, ctx)


@set_module('mxnet.numpy')
def split(ary, indices_or_sections, axis=0):
    """Split an array into multiple sub-arrays.
    Parameters
    ----------
    ary : ndarray
        Array to be divided into sub-arrays.
    indices_or_sections : int or 1-D array
        If `indices_or_sections` is an integer, N, the array will be divided
        into N equal arrays along `axis`.  If such a split is not possible,
        an error is raised.
        If `indices_or_sections` is a 1-D array of sorted integers, the entries
        indicate where along `axis` the array is split.  For example,
        ``[2, 3]`` would, for ``axis=0``, result in
          - ary[:2]
          - ary[2:3]
          - ary[3:]
        If an index exceeds the dimension of the array along `axis`,
        an empty sub-array is returned correspondingly.
    axis : int, optional
        The axis along which to split, default is 0.
    Returns
    -------
    sub-arrays : list of ndarrays
        A list of sub-arrays.
    Raises
    ------
    ValueError
        If `indices_or_sections` is given as an integer, but
        a split does not result in equal division."""
    return _mx_nd_np.split(ary, indices_or_sections, axis=axis)


@set_module('mxnet.numpy')
def concatenate(seq, axis=0, out=None):
    """Join a sequence of arrays along an existing axis.
    Parameters
    ----------
    a1, a2, ... : sequence of array_like
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).
    axis : int, optional
        The axis along which the arrays will be joined.  If axis is None,
        arrays are flattened before use.  Default is 0.
    out : ndarray, optional
        If provided, the destination to place the result. The shape must be
        correct, matching that of what concatenate would have returned if no
        out argument were specified.
    Returns
    -------
    res : ndarray
        The concatenated array.
    """
    return _mx_nd_np.concatenate(seq, axis=axis, out=out)


@set_module('mxnet.numpy')
def stack(arrays, axis=0, out=None):
    """Join a sequence of arrays along a new axis.
        The axis parameter specifies the index of the new axis in the dimensions of the result.
        For example, if `axis=0` it will be the first dimension and if `axis=-1` it will be the last dimension.
    Parameters
    ----------
    arrays : sequence of array_like
        Each array must have the same shape.
    axis : int, optional
        The axis in the result array along which the input arrays are stacked.
    out : ndarray, optional
        If provided, the destination to place the result. The shape must be correct,
        matching that of what stack would have returned if no out argument were specified.
    Returns
    -------
    stacked : ndarray
        The stacked array has one more dimension than the input arrays."""
    return _mx_nd_np.stack(arrays, axis=axis, out=out)