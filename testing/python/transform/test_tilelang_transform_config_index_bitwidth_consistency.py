import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tvm import tirx


def _apply(func, bits=None):
    config = {} if bits is None else {"tl.config_index_bitwidth": bits}
    with tilelang.transform.PassContext(config=config):
        result = tilelang.transform.ConfigIndexBitwidth()(tvm.IRModule({"main": func}))["main"]
    assert not tirx.analysis.undefined_vars(result.body, list(result.params))
    assert tirx.analysis.verify_well_formed(result)
    return result


def _nodes(stmt, node_type):
    result = []
    tirx.stmt_functor.post_order_visit(stmt, lambda node: result.append(node) if isinstance(node, node_type) else None)
    return result


@pytest.mark.parametrize("bits", [32, 64])
def test_config_index_bitwidth_preserves_parameters_and_metadata(bits):
    n, stride, offset, bias = [tirx.Var(name, "int32") for name in ("n", "stride", "offset", "bias")]
    source = tirx.decl_buffer((n,), "int32", name="A", strides=[stride], elem_offset=offset)
    output = tirx.decl_buffer((4,), "int32", name="B")
    i = tirx.Var("i", "int32")
    value = tirx.BufferLoad(source, [i]) + bias
    body = tirx.SeqStmt([tirx.DeclBuffer(source), tirx.For(i, 0, 4, tirx.ForKind.SERIAL, tirx.BufferStore(output, value, [i]))])
    func = tirx.PrimFunc([source.data, output.data, n, stride, offset, bias], body, buffer_map={source.data: source, output.data: output})
    after = _apply(func, bits)

    assert all(a.same_as(b) for a, b in zip(after.params, func.params))
    assert after.buffer_map[source.data].same_as(source)
    assert after.buffer_map[output.data].same_as(output)
    assert after.body.seq[0].buffer.same_as(source)
    loop = after.body.seq[1]
    assert loop.loop_var.same_as(i)
    assert loop.min.dtype == loop.extent.dtype == "int32"
    store = loop.body
    assert isinstance(store.value, tirx.Add)
    assert store.value.dtype == "int32"
    assert store.value.b.same_as(bias)
    assert store.indices[0].dtype == f"int{bits}"
    assert _nodes(store.value, tirx.BufferLoad)[0].indices[0].dtype == f"int{bits}"


@pytest.mark.parametrize("bits", [32, 64])
def test_config_index_bitwidth_preserves_thread_binding(bits):
    output = tirx.decl_buffer((4,), "int32", name="A")
    tx = tirx.Var("tx", "int32")
    iv = tirx.IterVar(tvm.ir.Range(0, 4), tx, tirx.IterVar.ThreadIndex, "threadIdx.x")
    body = tirx.AttrStmt(iv, "thread_extent", 4, tirx.BufferStore(output, tx, [tx]))
    after = _apply(tirx.PrimFunc([output.data], body, buffer_map={output.data: output}), bits)

    assert after.body.node.same_as(iv)
    assert after.body.value.dtype == "int32"
    assert after.body.body.value.same_as(tx)
    assert after.body.body.indices[0].dtype == f"int{bits}"


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("binding", ["bind", "let"])
def test_config_index_bitwidth_preserves_local_bindings(bits, binding):
    buffer = tirx.decl_buffer((8,), "int32", name="A")
    i, v = tirx.Var("i", "int32"), tirx.Var("v", "int32")
    if binding == "bind":
        body = tirx.SeqStmt([tirx.Bind(v, i + 1), tirx.BufferStore(buffer, v, [v])])
    else:
        body = tirx.BufferStore(buffer, i, [tirx.Let(v, i + 1, v + 1)])
    body = tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body)
    after = _apply(tirx.PrimFunc([buffer.data], body, buffer_map={buffer.data: buffer}), bits)

    node = _nodes(after.body, tirx.Bind if binding == "bind" else tirx.Let)[0]
    assert node.var.same_as(v)
    assert node.value.dtype == "int32"
    store = _nodes(after.body, tirx.BufferStore)[0]
    assert store.value.dtype == "int32"
    assert store.indices[0].dtype == f"int{bits}"


@pytest.mark.parametrize("kind", ["mul", "shift", "min", "conditional", "select"])
def test_config_index_bitwidth_widens_arithmetic_before_evaluation(kind):
    i = tirx.Var("i", "int32")
    expressions = {
        "mul": i * 4,
        "shift": i << 2,
        "min": tirx.min(i * 4, (1 << 31) - 1),
        "conditional": tirx.if_then_else(i < 1, 0, i * 4),
        "select": tirx.Select(i < 1, 0, i * 4),
    }
    buffer = tirx.decl_buffer((1,), "uint8", name="A")
    body = tirx.For(i, 0, 1 << 30, tirx.ForKind.SERIAL, tirx.Evaluate(tirx.BufferLoad(buffer, [expressions[kind]])))
    after = _apply(tirx.PrimFunc([buffer.data], body, buffer_map={buffer.data: buffer}), 64)
    index = _nodes(after.body, tirx.BufferLoad)[0].indices[0]

    assert index.dtype == "int64"
    assert after.body.loop_var.same_as(i)
    value = tirx.stmt_functor.substitute(index, {i: tirx.const(1 << 29, "int32")})
    expected = (1 << 31) - 1 if kind == "min" else 1 << 31
    assert tvm.arith.Analyzer().simplify(value).value == expected


@pytest.mark.parametrize("call_kind", ["extern", "clz"])
def test_config_index_bitwidth_keeps_call_abi_and_let_scope(call_kind):
    i, v = tirx.Var("i", "int32"), tirx.Var("v", "int32")
    buffer = tirx.decl_buffer((1,), "uint8", name="A")
    call = tirx.call_extern("int32", "index_from_i32", v) if call_kind == "extern" else tirx.call_intrin("int32", "tirx.clz", v)
    index = tirx.Let(v, i + 1, call)
    func = tirx.PrimFunc([buffer.data, i], tirx.Evaluate(tirx.BufferLoad(buffer, [index])), buffer_map={buffer.data: buffer})
    after = _apply(func, 64)

    call = _nodes(after.body, tirx.Call)[0]
    assert call.dtype == "int32"
    assert call.args[1 if call_kind == "extern" else 0].same_as(v)
    assert _nodes(after.body, tirx.Let)[0].var.same_as(v)
    assert _nodes(after.body, tirx.BufferLoad)[0].indices[0].dtype == "int64"


def test_config_index_bitwidth_keeps_explicit_cast_semantics():
    wide = tirx.Var("wide", "int64")
    buffer = tirx.decl_buffer((1,), "uint8", name="A")
    original_index = tirx.Cast("int32", wide)
    func = tirx.PrimFunc([buffer.data, wide], tirx.Evaluate(tirx.BufferLoad(buffer, [original_index])), buffer_map={buffer.data: buffer})
    after = _apply(func, 64)
    index = _nodes(after.body, tirx.BufferLoad)[0].indices[0]

    assert isinstance(index, tirx.Cast)
    assert index.dtype == "int64"
    assert index.value.same_as(original_index)


def test_config_index_bitwidth_keeps_index_buffer_element_type():
    i = tirx.Var("i", "int32")
    indices = tirx.decl_buffer((4,), "int32", name="indices")
    data = tirx.decl_buffer((16,), "uint8", name="data")
    body = tirx.For(i, 0, 4, tirx.ForKind.SERIAL, tirx.Evaluate(tirx.BufferLoad(data, [tirx.BufferLoad(indices, [i]) * 4])))
    after = _apply(tirx.PrimFunc([indices.data, data.data], body, buffer_map={indices.data: indices, data.data: data}), 64)

    loads = _nodes(after.body, tirx.BufferLoad)
    assert len(loads) == 2
    assert loads[0].buffer.same_as(indices)
    assert loads[0].dtype == "int32"
    assert all(load.indices[0].dtype == "int64" for load in loads)


def test_config_index_bitwidth_widens_vector_index_lanes():
    buffer = tirx.decl_buffer((1,), "uint8", name="A")
    index = tirx.Ramp((1 << 31) - 2, 1, 4)
    after = _apply(tirx.PrimFunc([buffer.data], tirx.Evaluate(tirx.BufferLoad(buffer, [index])), buffer_map={buffer.data: buffer}), 64)
    index = _nodes(after.body, tirx.BufferLoad)[0].indices[0]
    assert index.dtype == "int64x4"
    assert tvm.arith.Analyzer().simplify(index.base + index.stride * 3).value == (1 << 31) + 1


def test_config_index_bitwidth_widens_access_ptr_offset_only():
    i = tirx.Var("i", "int32")
    buffer = tirx.decl_buffer((16,), "uint8", name="A")
    ptr = tirx.call_intrin("handle", "tirx.tvm_access_ptr", tirx.type_annotation("uint8"), buffer.data, i * 2, 1, 1)
    after = _apply(tirx.PrimFunc([buffer.data, i], tirx.Evaluate(ptr), buffer_map={buffer.data: buffer}), 64)
    call = _nodes(after.body, tirx.Call)[-1]
    assert call.args[1].same_as(buffer.data)
    assert call.args[2].dtype == "int64"
    assert call.args[3].dtype == call.args[4].dtype == "int32"


@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_config_index_bitwidth_32_preserves_automatic_legalization(index_dtype):
    i = tirx.Var("i", index_dtype)
    buffer = tirx.decl_buffer((1,), "uint8", name="A")
    body = tirx.For(
        i,
        tirx.const(0, index_dtype),
        tirx.const(1 << 30, index_dtype),
        tirx.ForKind.SERIAL,
        tirx.Evaluate(tirx.BufferLoad(buffer, [i * 4])),
    )
    func = tirx.PrimFunc([buffer.data], body, buffer_map={buffer.data: buffer})
    after = _apply(func, 32)
    tvm.ir.assert_structural_equal(after, _apply(func))
    assert _nodes(after.body, tirx.BufferLoad)[0].indices[0].dtype == "int64"
    assert after.body.loop_var.same_as(i)


@pytest.mark.parametrize("bits", [-1, 0, 16, 128])
def test_config_index_bitwidth_rejects_unsupported_width(bits):
    func = tirx.PrimFunc([], tirx.Evaluate(0))
    with pytest.raises(ValueError, match="tl.config_index_bitwidth.*32.*64"):
        _apply(func, bits)


@pytest.mark.parametrize(
    "target",
    [
        pytest.param("llvm", marks=tilelang.testing.requires_llvm.marks()),
        pytest.param("cuda", marks=tilelang.testing.requires_cuda.marks()),
        pytest.param("hip", marks=tilelang.testing.requires_rocm.marks()),
    ],
)
@pytest.mark.parametrize("bits", [32, 64])
def test_config_index_bitwidth_dynamic_transpose(target, bits):
    m, n = T.dynamic("m n")

    @T.prim_func
    def main(A: T.Tensor((m, n), T.int32), B: T.Tensor((n, m), T.int32)):
        with T.Kernel(T.ceildiv(m * n, 128), threads=128) as bx:
            for tx in T.Parallel(128):
                idx = bx * 128 + tx
                if idx < m * n:
                    row = idx // n
                    col = idx % n
                    B[col, row] = T.min(A[row, col] * 4, T.int32((1 << 31) - 1))

    kernel = tilelang.compile(
        main,
        target=target,
        execution_backend="tvm_ffi",
        pass_configs={
            "tl.config_index_bitwidth": bits,
            "tl.disable_warp_specialized": True,
        },
    )
    device = "cpu" if target == "llvm" else "cuda"
    for rows, cols in [(1, 1), (3, 7), (17, 9)]:
        a = torch.arange(rows * cols, dtype=torch.int32, device=device).reshape(rows, cols)
        a[0, 0] = (1 << 29) - 1
        if rows * cols > 1:
            a[-1, -1] = -(1 << 29)
        b = torch.empty((cols, rows), dtype=torch.int32, device=device)
        kernel(a, b)
        # Data arithmetic remains int32 even when address arithmetic is int64.
        expected = torch.minimum(a * 4, torch.full_like(a, (1 << 31) - 1)).t()
        torch.testing.assert_close(b, expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
