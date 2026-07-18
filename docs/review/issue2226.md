# Issue #2226: CPU Fallback Thread Placeholder Cleanup Plan

## 1. 背景与目标

Issue: [tile-ai/tilelang#2226](https://github.com/tile-ai/tilelang/issues/2226)

CPU `c`/`llvm` lowering 当前复用了一批面向 SIMT thread 的 layout 和 tile-op helper。由于 CPU pipeline 中不存在真实的 `threadIdx.x`，`LayoutInference` 和 `LowerTileOp` 各自创建了一个 extent 为 1、名字为 `v_thread` 的 synthetic `Var`，将它当作逻辑 thread index 使用。

当前实现依靠 `LowerTileOp` 末尾的兼容清理，将这个 synthetic `Var` 改写为常量 `0`。该清理解决了原始编译失败，但仍有以下问题：

- synthetic `Var` 在被清理前已经进入 TIR expression 或 annotation；
- 两个 pass 创建的是不同 identity 的 `Var`，只能通过 `name_hint` 兼容匹配；
- 名称匹配可能错误改写合法的同名变量；
- 正确性依赖一个容易被移动、遗漏或绕过的 late cleanup；
- `LowerTileOp` 与 helper API 混淆了线程绑定、线程范围和逻辑线程索引三个不同概念。

最终目标是：

1. CPU lowering 不再创建任何裸 synthetic thread `Var`；
2. CPU helper 从第一次消费 logical thread index 开始就接收常量 `0`；
3. GPU helper 继续接收受 `thread_extent` 绑定的真实 `threadIdx.x`；
4. `LowerTileOp` 删除 CPU fallback canonicalizer；
5. `SplitHostDevice`、`MakePackedAPI` 以及后续 pass 永远看不到 synthetic thread placeholder；
6. 合法的同名 `v_thread` 参数、循环变量或局部变量不受影响。

本文只描述修复设计、修改范围、PR 拆分和验证方法，不包含实现。

## 2. 现状

### 2.1 CPU kernel launch 已经没有真实 thread binding

CPU pipeline 在 `tilelang/cpu/pipeline.py` 中按以下顺序执行关键 pass：

```text
BindTarget
  -> MaterializeKernelLaunch(lower_thread_binding=False)
  -> ...
  -> LayoutInference
  -> LowerTileOp
  -> ...
  -> AnnotateDeviceRegions
  -> SplitHostDevice
  -> MakePackedAPI
```

`MaterializeKernelLaunch(lower_thread_binding=False)` 会把：

- `blockIdx.*` 转换为普通 serial loop；
- `threadIdx.*` 转换为 extent 为 1 的 serial loop，使原 kernel launch loop variable 在其 body 内固定为 0；
- 不生成 `AttrStmt(thread_extent)`。

因此，CPU pass 后续不会发现真实的 `threadIdx.x` `IterVar`。需要特别区分：launch materialization 保留的 unit serial loop variable 是原 kernel launch variable，而 issue #2226 中泄漏的 `v_thread` 是 `LayoutInference`/`LowerTileOp` 额外创建的、没有任何 TIR binding 的另一个变量。

相关位置：

- `tilelang/cpu/pipeline.py:15-35`
- `src/transform/materialize_kernel_launch.cc:5-20`
- `src/transform/materialize_kernel_launch.cc:80-94`

### 2.2 `LayoutInference` 创建第一份 fallback

`BufferUseDefCollector` 当前包含默认成员：

```cpp
IterVar thread_var_ = IterVar(
    Range::FromMinExtent(0, 1),
    Var("v_thread"),
    IterVarType::kDataPar);
```

相关位置：

- `src/transform/layout_inference.cc:1015-1019`

如果 visitor 遇到真实 `AttrStmt(thread_extent)` 且 thread tag 是 `threadIdx.x`，该成员会被真实 `IterVar` 替换：

- `src/transform/layout_inference.cc:802-810`

CPU pipeline 不存在该 attribute，因此始终使用默认 fallback。

在收集 tile operator/parallel loop 时，pass 同时保存：

- `thread_var_vec_`：当前 `IterVar`；
- `thread_bounds_vec_`：`ComputeThreadBounds(...)` 的结果；
- 当前 analyzer snapshot。

相关位置：

- `src/transform/layout_inference.cc:550-572`
- `src/transform/layout_inference.cc:739-744`

layout inference 本身主要需要的是 `Range thread_bounds`。`LayoutInferArgs` 已经只携带 `thread_bounds`，没有携带 runtime thread `Var`：

- `src/op/operator.h:131-144`

但 parallel predicate materialization 会将 `thread_var->var` 代入 predicate placeholder：

```cpp
for_infer->GetPredicate(thread_var->var)
```

然后把结果写入 `parallel_loop_predicate` annotation：

- `src/transform/layout_inference.cc:451-479`
- `src/transform/layout_inference.cc:1276-1283`
- `src/op/parallel.h:128-129`
- `src/op/parallel.cc:583-589`

这使 `LayoutInference` 创建的 fallback `Var` 进入可被后续 pass 消费的 IR metadata。annotation payload 并不保证被通用 expression mutator 遍历，因此不能依赖更晚的通用 rewrite 修复它。

另外，`RunInferStep` 当前要求每个 `thread_var_vec_` 元素必须存在，并通过 `iter_var->dom->extent` 检查线程 extent 是否为常量：

- `src/transform/layout_inference.cc:127-151`

该检查实际需要的是已经单独保存的 `thread_bounds->extent`，不需要一个 synthetic `IterVar`。

`FloatingBufferCollector` 也单独维护 `IterVar thread_var_` 来计算 floating fragment 的 thread bounds：

- `src/transform/layout_inference.cc:931-982`

该路径同样只需要 bounds；在无真实 binding 时返回 `[0, 1)` 即可。

### 2.3 `LowerTileOp` 创建第二份 fallback

`LowerTileOpPass` 当前也包含一份独立的默认成员：

```cpp
IterVar thread_var_ = IterVar(
    Range::FromMinExtent(0, 1),
    Var("v_thread"),
    IterVarType::kDataPar);
```

相关位置：

- `src/transform/lower_tile_op.cc:1531-1540`

如果遇到真实 `threadIdx.x` `thread_extent`，该成员会被真实 `IterVar` 替换：

- `src/transform/lower_tile_op.cc:1225-1235`

CPU pipeline 中没有该 attribute，因此 `LowerTileOp` 使用自己的 fallback。它与 `LayoutInference` 的 fallback 名称相同，但不是同一个 `Var` 对象。

`LowerTileOp` 通过两条路径把 fallback 交给 lowering helper：

1. `LowerArgs.thread_var`

   ```cpp
   lower_args.thread_bounds = CurrentThreadBounds();
   lower_args.thread_var = thread_var_->var;
   ```

   相关位置：

   - `src/op/operator.h:95-103`
   - `src/transform/lower_tile_op.cc:1142-1222`

2. parallel-loop lowering

   ```cpp
   LowerParallelLoop(..., thread_var_->var, ...)
   ```

   相关位置：

   - `src/transform/lower_tile_op.cc:1498-1506`

`LowerArgs.thread_var` 当前类型为 `tirx::Var`，但绝大多数消费者只把它用于：

- 算术表达式；
- predicate substitution；
- layout forward/inverse；
- 与 thread range min 比较；
- 构造 leader-thread guard；
- 传入 Python GEMM/GEMM-SP lowering。

这些用途需要的是 `PrimExpr`，不要求变量 identity。

### 2.4 Loop partition 对 `Var` 的约束也是表示问题

`PartitionLoop` 和 `LowerParallelLoop` 当前把 thread 参数声明为 `Var`：

- `src/transform/loop_partition.h:40-42`
- `src/transform/loop_partition.h:72-77`
- `src/transform/loop_partition.cc:66-98`
- `src/transform/loop_partition.cc:274-311`

`PartitionLoop` 将 output loop vars 与 thread var 一起传给 inverse layout。inverse layout 只消费 expression，因此输入本应是 `Array<PrimExpr>`。

当前代码还通过：

```cpp
Map<Var, PrimExpr> thread_offset_map;
thread_offset_map.Set(thread_var, thread_var - range->min);
```

处理非零 thread range min，并在 indices、replicate index 和最终 body 上执行 substitution：

- `src/transform/loop_partition.cc:90-97`
- `src/transform/loop_partition.cc:127-166`

这迫使 thread index 必须是 `Var`。更自然的表示是在送入 inverse layout 前直接构造 normalized expression：

```cpp
PrimExpr normalized_thread_index = thread_index - thread_range->min;
```

后续 inverse indices、guard 和 replicate index 全部基于同一个 expression，不再需要以 thread `Var` 为 key 的 substitution map。

### 2.5 少数调用点确实使用了变量 identity

绝大多数调用点可直接从 `Var` 泛化到 `PrimExpr`，但有两个需要显式处理的特例。

#### 普通 copy 的 thread dependency 判断

`src/op/copy.cc:40-60` 当前用：

```cpp
v == lower_args.thread_var.get()
```

判断 destination range 是否依赖 thread。改为 `PrimExpr` 后不能再假设 logical index 本身是单个 `Var`。

正确处理方式是按 identity 收集 `thread_index` 使用的变量，并检查 destination range 是否使用其中任意变量：

- GPU `thread_index = threadIdx.x` 时保持当前语义；
- CPU `thread_index = 0` 时变量集合为空，明确表示不依赖线程；
- 将来若传入 normalized expression，也不会错误 downcast 或按名称匹配。

#### CUDA LDSM/thread offset normalization

`src/cuda/op/copy.cc:1201-1207` 当前把 `lower_args.thread_var` 作为 substitution map key。改为 `PrimExpr` 后，应在构造相关索引时直接使用 normalized thread expression，不能将任意 expression 强制转换回 `Var`。

### 2.6 当前 canonicalizer 的行为与风险

`LowerTileOp` 末尾包含：

```cpp
class CPUFallbackThreadVarCanonicalizer : public StmtExprMutator {
  ...
  if (op == fallback_thread_var_.get() ||
      op->name_hint == fallback_thread_var_->name_hint) {
    return make_zero(op->dtype);
  }
};
```

相关位置：

- `src/transform/lower_tile_op.cc:192-220`
- `src/transform/lower_tile_op.cc:322-328`

identity 匹配只能删除 `LowerTileOp` 自己创建的 fallback。名称匹配用于删除 `LayoutInference` 创建的另一份 fallback。

该策略有以下风险：

1. 任意合法变量只要名为 `v_thread`，都可能被静默改写为 0；
2. cleanup 是否成功依赖两个 pass 继续使用同一 `name_hint`；
3. CPU 判断使用 `TargetIsCPU`，实际覆盖 `c` 和 `llvm`，并不只覆盖注释中的 CPU `c`；
4. 如果 placeholder 出现在 mutator 不遍历的 metadata 中，cleanup 可能遗漏；
5. 如果 cleanup 被重排到 `SplitHostDevice` 之后，kernel ABI 已经被污染；
6. 即使编译成功，也无法证明 placeholder 没有被错误当作真实参数处理。

### 2.7 为什么会污染 kernel ABI

`SplitHostDevice` 对 device body 运行 `VarUseDefAnalyzer`：

- `src/transform/split_host_device.cc:401-425`

所有未定义 `Var` 会进入 device function 参数列表。随后 pass 会为 device side 创建新参数，并在 host launch 中使用原始未定义变量。

因此 leaked fallback 会导致：

- device kernel 获得意外的 scalar `v_thread` ABI 参数；
- host launch 使用未绑定的 `v_thread`。

`MakePackedAPI` 最后检查 host PrimFunc body 中是否还有不在参数列表中的变量：

- `src/transform/make_packed_api.cc:653-657`

这正是原问题最终报错的位置。把 cleanup 放到 `SplitHostDevice` 或 `MakePackedAPI` 中只会隐藏上游表示错误，不能作为最终方案。

## 3. 根本原因

根本原因不是 `T.While`，也不是 `SplitHostDevice` 的 free-variable 分析过于严格，而是 lowering API 把三个独立概念错误地压缩成了一个 `IterVar/Var`：

| 概念 | 实际用途 | 正确类型 | CPU 值 | GPU 值 |
| --- | --- | --- | --- | --- |
| thread binding | 表示真实 `thread_extent` 绑定及变量 identity | `Optional<IterVar>` | 未定义 | 真实 `threadIdx.x` `IterVar` |
| thread bounds | layout inference、range analysis、thread extent | `Range` | `[0, 1)` | 真实线程范围 |
| logical thread index | helper 中的算术、predicate、layout inverse、guard | `PrimExpr` | 常量 `0` | 真实 `threadIdx.x` `Var` |

`Range(0,1)` 不会在 TIR 中定义一个变量。把 `Var("v_thread")` 放进 `IterVar` 只给 C++ 对象附加了 domain metadata，并没有产生以下任何一种合法 binding：

- `For` loop variable；
- `AttrStmt(thread_extent)`；
- `LetStmt`/`Bind`；
- PrimFunc parameter。

因此，只要该 `Var` 进入生成 TIR，它就是自由变量。

两个 pass 又各自创建相同名字、不同 identity 的对象，使 identity-safe cleanup 无法覆盖两条路径，最终引入了不安全的 name-based rewrite。

## 4. 完整修复方案

### 4.1 目标数据模型

采用 `PrimExpr logical thread index` 方案，并明确保留 bounds 与 binding：

```text
真实 thread binding: Optional<IterVar>
thread bounds:        Range
logical thread index: PrimExpr
```

CPU：

```text
thread_binding = nullopt
thread_bounds  = [0, 1)
thread_index   = IntImm(0)
```

GPU：

```text
thread_binding = real threadIdx.x IterVar
thread_bounds  = ComputeThreadBounds(thread_binding, analyzer)
thread_index   = thread_binding->var
```

`IntImm(0)` 的 dtype 必须与 thread bounds/index arithmetic 使用的 dtype 一致。实现时应从当前 binding 或 `thread_bounds->min.dtype()` 派生，而不是在多个调用点散落硬编码类型。

不建议当前就引入公开的 `ThreadContext` struct。它在概念上完整，但会扩大每个 tile operator、C++/Python FFI 和 backend callback 的改动。当前 `LowerArgs` 已经单独携带 `thread_bounds`，把 `thread_var` 改为 `PrimExpr thread_index` 就足以修复错误抽象。

### 4.2 泛化 helper API

先完成行为保持的 API 重构：

1. `LowerArgs::thread_var` 改名并改型为：

   ```cpp
   PrimExpr thread_index;
   ```

2. `ParallelOpNode::GetPredicate(Var)` 改为：

   ```cpp
   Optional<PrimExpr> GetPredicate(PrimExpr thread_index) const;
   ```

3. `PartitionLoop` 和 `LowerParallelLoop` 的 thread 参数改为 `PrimExpr`。

4. inverse layout 输入使用 `Array<PrimExpr>`。

5. thread range normalization 直接生成 normalized `PrimExpr`，不再用 thread `Var` 作为 substitution key。

6. 所有 common/backend operator 调用点更新字段名和签名；表达式消费者不做 `Downcast<Var>`。

7. copy dependency analysis 按变量 identity 集合判断 expression dependency。

8. CUDA LDSM 路径直接使用 normalized expression。

9. GEMM/GEMM-SP C++/Python lowering 参数及 type hint 从 `tirx.Var` 改为 `tirx.PrimExpr`，但不改变 FFI 参数顺序、注册名或 runtime ABI。

这一阶段仍可由 `LayoutInference`/`LowerTileOp` 传入现有真实或 fallback `Var`，因此应当不改变生成语义。

### 4.3 移除 `LayoutInference` fallback

`BufferUseDefCollector` 中不再默认创建 `IterVar(Var("v_thread"))`。

建议内部状态：

```cpp
Optional<IterVar> thread_binding_;
std::vector<PrimExpr> thread_index_vec_;
std::vector<Range> thread_bounds_vec_;
```

收集每个 operator 时：

- 有真实 binding：保存真实 `thread_binding_->var`；
- 无真实 binding：保存 dtype 正确的常量 0；
- bounds 始终单独保存。

也可以保存 `Optional<IterVar>` per-op，但如果后续唯一用途是取得 index expression，则直接保存 `PrimExpr` 更清晰。实现前应以所有 `thread_var_vec_` 调用点为准，避免保留无用途的 identity state。

`CurrentThreadBounds()` 修改为：

- 有真实 binding 时调用 `ComputeThreadBounds`；
- 无 binding 时返回 `[0,1)`。

`RunInferStep` 改为直接验证 `thread_bounds_vec_[i]->extent` 是常量，不再要求 fallback `IterVar` defined。

parallel predicate materialization 使用 per-op `thread_index`：

```cpp
for_infer->GetPredicate(thread_index)
```

CPU annotation 在写入时就已经是基于 0 的表达式，不依赖后续 mutator 清理。

`FloatingBufferCollector` 同样改为 `Optional<IterVar>`，无 binding 时直接返回 `[0,1)`；该 collector 不需要 logical thread index。

完成这一步后，`LayoutInference` 中不再出现 `Var("v_thread")`。

### 4.4 过渡期将 cleanup 收紧为 identity-only

移除 `LayoutInference` fallback 后，`LowerTileOp` canonicalizer 不再需要通过名称匹配另一个 pass 创建的变量。

在最终删除 canonicalizer 前，先将条件收紧为：

```cpp
op == fallback_thread_var_.get()
```

删除：

```cpp
op->name_hint == fallback_thread_var_->name_hint
```

这样中间版本仍可正确清理 `LowerTileOp` 自己的 fallback，同时立即消除合法同名变量被错误改写的风险。

此过渡步骤必须与 `LayoutInference` fallback 移除在同一个 PR 中完成。否则只做 identity-only cleanup 会让 LayoutInference 创建的另一份 fallback 重新泄漏。

### 4.5 移除 `LowerTileOp` fallback

将 `LowerTileOpPass` 默认 synthetic `IterVar` 改为：

```cpp
Optional<IterVar> thread_binding_;
```

只在 visitor 遇到真实 `threadIdx.x` `thread_extent` 时设置。

增加两个职责清晰的内部 helper：

```cpp
Range CurrentThreadBounds() const;
PrimExpr CurrentThreadIndex() const;
```

语义：

- 有真实 binding：返回真实 bounds 和真实 `Var`；
- 无真实 binding：返回 `[0,1)` 和常量 0。

在同一个 lowering scope 中先计算一次 bounds/index，再用于：

- `LowerArgs.thread_bounds`；
- `LowerArgs.thread_index`；
- `LowerParallelLoop`；
- predicate、copy、layout inverse 和 backend helper。

这样可以避免不同调用点独立构造零值时产生 dtype 不一致。

实现不能只用 `TargetIsCPU(target_)` 判断 index 是否为 0。是否存在真实 binding 是表示事实：

- 普通 CPU pipeline 无 binding，自然得到 0；
- GPU pipeline 有 binding，使用真实 `Var`；
- 如果某个 CPU PrimFunc 显式携带真实 `thread_extent`，不应由 broad CPU cleanup 静默改写，应该尊重显式 binding，或在不支持该输入时给出明确诊断。

### 4.6 删除 compatibility cleanup

当两条 fallback 来源都被移除后，删除：

- `CPUFallbackThreadVarCanonicalizer`；
- `LowerTileOpPass::Substitute` 末尾的 CPU rewrite；
- `TODO(#2226)`；
- 两处 synthetic fallback 字段及 workaround 注释。

不在以下 pass 添加任何替代兜底：

- `SplitHostDevice`；
- `MakePackedAPI`；
- `Simplify`；
- codegen。

最终 invariant 是 synthetic thread `Var` 从未被创建或写入 lowered TIR，而不是在 pass 末尾被扫描删除。

### 4.7 thread scope 状态管理

当前 `LayoutInference` 和 `LowerTileOp` 都通过成员保存最近遇到的 `threadIdx.x`，没有显式 scope stack。实施时需要检查 visitor 的作用域语义：

- 如果 IR 保证每个 PrimFunc 只有一个包围主体的 `threadIdx.x` attribute，保持当前模型即可；
- 如果允许 sibling/nested `thread_extent`，visitor 进入 scope 时必须保存旧 binding，访问 body 后恢复，避免一个 scope 的 binding 污染后续 sibling；
- 相应测试应覆盖至少两个 sibling thread scopes 或明确记录该结构不被前置 pass 生成。

该问题不是 #2226 的直接根因，但把成员改为 `Optional<IterVar>` 时应避免固化现有潜在 scope bug。

## 5. 需要修改的范围

以下范围分为确定修改和验证后可能修改。实际实现以编译器报错和全仓 `rg` 结果为准，不应机械修改其他 pass 中语义上确实要求真实 `IterVar` 的 `thread_var`。

### 5.1 Core API 与 shared transform

确定修改：

| 文件 | 修改内容 |
| --- | --- |
| `src/op/operator.h` | `LowerArgs.thread_var: Var` 改为 `thread_index: PrimExpr` |
| `src/op/parallel.h` | `GetPredicate` 参数改为 `PrimExpr` |
| `src/op/parallel.cc` | predicate substitution 接受任意 `PrimExpr` |
| `src/transform/loop_partition.h` | `PartitionLoop`/`LowerParallelLoop` 参数改为 `PrimExpr` |
| `src/transform/loop_partition.cc` | inverse input 和 thread offset normalization 改为 expression-based |
| `src/transform/layout_inference.cc` | 移除 fallback；分离 optional binding、bounds、index；CPU predicate 直接用 0 |
| `src/transform/lower_tile_op.cc` | 移除 fallback；传 logical index；最终删除 canonicalizer |
| `src/op/copy.cc` | 泛化 dependency analysis；更新 helper 调用 |

不应修改：

- 其他 transform 中确实用于识别/绑定真实 thread scope 的 `IterVar thread_var_`，例如 CUDA shared barrier/tmem lowering；
- `ThreadBindingCollector` 等以 binding identity 为目的的 helper；
- `LayoutReducer` 中真实 thread binding 的状态，除非编译或测试证明它共享了该错误 fallback。

### 5.2 Common operator lowering

需要更新 `LowerArgs` 字段引用和 helper 签名，主要包括：

- `src/backend/common/op/fill.h`
- `src/backend/common/op/transpose.h`
- `src/backend/common/op/atomic_reduce.h`
- `src/backend/common/op/reduce.h`
- `src/op/copy.cc`
- `src/op/gemm.cc`
- `src/op/gemm_sp.cc`

这些路径大部分只做 expression arithmetic 或 predicate/layout substitution。

### 5.3 Backend operator lowering

通过全仓搜索确认并更新：

- CUDA
  - `src/cuda/op/copy.cc`
  - `src/cuda/op/fill.cc`
  - `src/cuda/op/atomic_add.cc`
  - 其他直接访问 `LowerArgs.thread_var` 的文件
- ROCm
  - `src/rocm/op/copy.cc`
  - `src/rocm/op/atomic_add.cc`
- Metal
  - `src/metal/op/copy.cc`
  - `src/metal/op/fill.cc`
  - `src/metal/op/transpose.cc`
- WebGPU
  - `src/webgpu/op/fill.cc`
  - `src/webgpu/op/transpose.cc`

修改原则：

- 算术和比较直接使用 `PrimExpr thread_index`；
- 不为满足旧写法把 expression 重新包装成 synthetic `Var`；
- 不使用 `name_hint` 判断 thread identity；
- 真实需要 identity 的逻辑通过 expression 中变量集合处理，或明确要求真实 binding 的独立 API。

### 5.4 Python GEMM/GEMM-SP lowering

至少检查并更新：

- `tilelang/tileop/gemm/__init__.py`
- `tilelang/tileop/gemm/gemm_base.py`
- `tilelang/tileop/gemm_sp/__init__.py`
- `tilelang/tileop/gemm_sp/gemm_sp_base.py`
- `tilelang/tileop/gemm_sp/gemm_sp_wgmma.py`
- 其他 backend GEMM implementation 中声明 `thread_var: tirx.Var` 的 type hint

字段/参数建议统一改名为 `thread_index`。如果一次改名会影响过多 Python subclass keyword 参数，可以在 PR 1 中先改类型、保持参数名，再在同一 PR 的独立 commit 完成一致性改名；最终不应继续暴露错误的 `Var` 类型约束。

### 5.5 测试范围

建议修改或新增：

| 文件 | 覆盖目标 |
| --- | --- |
| `testing/python/transform/test_tilelang_transform_lower_tile_op.py` | CPU pass boundary、无 synthetic var、同名变量不被改写 |
| `testing/python/transform/test_tilelang_transform_layout_inference.py` | CPU predicate 直接使用 0；GPU predicate 保留真实 binding |
| `testing/python/transform/test_tilelang_transform_split_host_device.py` | split 后 device ABI 无 synthetic thread 参数 |
| `testing/python/cpu/test_tilelang_cpu_while_repro.py` | C target 端到端原始回归 |
| `testing/python/llvm/test_tilelang_llvm_while_repro.py` | LLVM target 共用 CPU pipeline 回归 |

视 helper 可测试性和现有测试结构，增加 loop partition non-zero thread range min 的直接测试。它应验证：

- inverse indices 使用一次且仅一次 offset normalization；
- bounds guard 与 replicate index 使用同一个 normalized thread expression；
- GPU `threadIdx.x` identity 未改变。

## 6. 建议的 PR 拆分方案

建议拆为三个顺序依赖的 PR：

```text
PR 1: helper API 接受 PrimExpr，保持行为不变
  -> PR 2: LayoutInference 不再创建 fallback，cleanup 收紧为 identity-only
    -> PR 3: LowerTileOp 不再创建 fallback，删除 cleanup
```

该拆分的核心原则是：每个 PR 都能独立构建、测试、合入和回滚；在最后一个 fallback 来源消失之前，不提前删除必要 cleanup。

### 6.1 PR 1: Generalize logical thread index APIs to `PrimExpr`

建议标题：

```text
[Refactor] Generalize logical thread index APIs to PrimExpr
```

内容：

1. `LowerArgs.thread_var` 改为 `PrimExpr thread_index`；
2. `ParallelOpNode::GetPredicate` 改为接受 `PrimExpr`；
3. `PartitionLoop`/`LowerParallelLoop` 改为接受 `PrimExpr`；
4. expression-based thread range normalization；
5. 更新 common/CUDA/ROCm/Metal/WebGPU consumers；
6. 适配 copy dependency check 和 CUDA LDSM 特例；
7. 更新 GEMM/GEMM-SP C++/Python FFI type hint 和参数命名；
8. 增加 expression thread-index 与 non-zero range regression。

本 PR 中：

- `LayoutInference` 仍可传现有 fallback `Var`；
- `LowerTileOp` 仍可传现有 fallback `Var`；
- canonicalizer 保持不变；
- CPU/GPU 生成行为应保持不变。

PR 1 correctness gate：

- 所有声明和调用点完成类型迁移；
- 不存在对 `PrimExpr` 的不安全 `Downcast<Var>`；
- non-zero thread min 不重复 offset；
- CPU while regression继续通过；
- 可用 GPU backend 的 copy/layout tests 继续通过。

建议 commits：

1. `Refactor thread-index helper signatures to PrimExpr`
2. `Adapt operator and backend thread-index consumers`
3. `Add expression thread-index regression coverage`

如果签名修改与调用点拆开会导致中间 commit 无法编译，应合并为一个可构建 commit。保持 bisect 正确性优先于形式上的 commit 粒度。

### 6.2 PR 2: Remove the `LayoutInference` fallback

建议标题：

```text
[Transform] Avoid synthetic CPU thread vars in LayoutInference
```

内容：

1. `LayoutInference` 使用 `Optional<IterVar>` 表示真实 binding；
2. per-op 单独记录 logical `PrimExpr thread_index` 和 `Range thread_bounds`；
3. CPU bounds 为 `[0,1)`；
4. CPU parallel predicate 在写 annotation 前直接代入 0；
5. `RunInferStep` 验证 bounds extent，不再验证 fallback IterVar；
6. `FloatingBufferCollector` 无 binding 时使用 `[0,1)`；
7. 删除 `LayoutInference` 中的 `Var("v_thread")`；
8. 将临时 `LowerTileOp` canonicalizer 收紧为 identity-only；
9. 增加 CPU predicate、GPU identity 和合法同名变量测试。

为什么 identity-only cleanup 必须放在此 PR：

- 在移除 LayoutInference fallback 前收紧 cleanup，会遗漏另一个 identity 并恢复原始泄漏；
- 在移除后保留 name matching，则仍存在合法同名变量误写风险；
- 两者同 PR 能保证合入点始终正确。

PR 2 correctness gate：

- `LayoutInference` 不再构造 synthetic `v_thread`；
- CPU annotation 中没有 synthetic thread var；
- GPU annotation 引用真实受绑定的 thread var；
- 合法同名 `v_thread` 不被 canonicalizer 改写；
- `LowerTileOp` 自己的 fallback 仍由 identity cleanup 处理；
- C/LLVM while regression继续通过。

建议 commits：

1. `Represent absent LayoutInference thread binding explicitly`
2. `Materialize CPU layout predicates with zero`
3. `Restrict temporary fallback cleanup to variable identity`
4. `Add CPU predicate and name-collision regressions`

如果 commit 2 与 3 单独产生不正确中间状态，应合并。每个可推送 commit 都必须保持 fallback 不泄漏。

### 6.3 PR 3: Remove the `LowerTileOp` fallback and cleanup

建议标题：

```text
[Transform] Remove CPU fallback thread placeholder from LowerTileOp
```

内容：

1. `LowerTileOpPass` 使用 `Optional<IterVar>` 表示真实 binding；
2. 实现 `CurrentThreadBounds()` 和 `CurrentThreadIndex()`；
3. CPU `LowerArgs.thread_index` 直接为 0；
4. CPU `LowerParallelLoop` 直接接收 0；
5. GPU 路径继续使用真实 `threadIdx.x`；
6. 删除 `LowerTileOp` synthetic `Var("v_thread")`；
7. 删除 `CPUFallbackThreadVarCanonicalizer` 和 TODO #2226；
8. 增加 LowerTileOp pass-boundary、SplitHostDevice ABI 和端到端回归。

PR 3 correctness gate：

- 两个 pass 都不再创建 synthetic thread `Var`；
- `LowerTileOp` 中不存在 fallback cleanup；
- `LowerTileOp` 输出无 synthetic free var；
- split 后 device ABI 无 synthetic scalar 参数；
- host/device 参数与 launch 对应；
- C 与 LLVM 原始回归通过；
- 可用 GPU backend 的真实 `threadIdx.x` 行为无回归。

建议 commits：

1. `Represent absent LowerTileOp thread binding explicitly`
2. `Pass zero as the CPU logical thread index`
3. `Remove CPU fallback canonicalization`
4. `Add pass-boundary and device ABI regressions`

前 3 项如果无法形成各自正确且可构建的中间状态，应合并。尤其不能先删除 canonicalizer，再提交 direct-zero 路径。

### 6.4 不建议的拆分

以下拆分不建议采用：

1. 先删除 canonicalizer，再修两个 producer
   - 中间版本会恢复 free-var/ABI 问题。

2. 单独提交 identity-only cleanup，再移除 LayoutInference fallback
   - 中间版本无法清理 LayoutInference 的另一个 identity。

3. 把 cleanup 移到 `SplitHostDevice` 或 `MakePackedAPI`
   - 会让通用 ABI pass 承担 TileLang-specific 表示修复；
   - 可能只隐藏 host error，而 device ABI 已经被污染。

4. 只共享一个 synthetic `Var` identity
   - 可以删除 name matching，但仍会 materialize 一个无 binding 的自由变量；
   - correctness 仍依赖 late cleanup，没有解决根因。

5. 只把 `LowerTileOp` 参数改为 0
   - `LayoutInference` predicate annotation 仍可能携带另一份 fallback。

## 7. 正确性不变量

整个实施过程中，每个 PR 都必须保持以下 invariant：

1. CPU logical thread index 的语义恒为 0；
2. CPU thread bounds 恒为 `[0,1)`；
3. bounds 只是分析域，不能被当作变量 binding；
4. GPU logical thread index 是受 `thread_extent` 绑定的真实 `Var`；
5. GPU bounds 继续来自真实 `IterVar`/analyzer；
6. 不按 `name_hint` 判断 thread identity；
7. annotation 中不保存未绑定 synthetic `Var`；
8. non-zero thread range min 在 inverse layout、guard 和 replicate index 中使用相同 normalization，且只应用一次；
9. `SplitHostDevice` 和 `MakePackedAPI` 不承担 fallback 修复；
10. 合法同名变量不能因本修复改变语义；
11. 每个 PR/commit 在计划合入点都能构建并通过其对应测试，不依赖后续 PR 才恢复正确性。

## 8. 验证方式

### 8.1 静态检查

每个 PR 完成后执行全仓搜索。

PR 1 后：

```bash
rg -n "LowerArgs|thread_var|thread_index" src tilelang
rg -n "GetPredicate\(|LowerParallelLoop\(|PartitionLoop\(" src
```

确认：

- `LowerArgs.thread_var` 已全部迁移；
- 新 API 不再要求 `Var`；
- 未误改其他语义上需要真实 `IterVar` 的 transform。

PR 2 后：

```bash
rg -n 'Var\("v_thread"\)|name_hint.*v_thread|v_thread.*name_hint' \
  src/transform/layout_inference.cc src/transform/lower_tile_op.cc
```

确认：

- `LayoutInference` 不再创建 fallback；
- canonicalizer 不再按名称匹配；
- `LowerTileOp` 暂时只剩自身 fallback identity cleanup。

PR 3 后：

```bash
rg -n 'Var\("v_thread"\)|CPUFallbackThreadVarCanonicalizer|TODO\(#2226\)' \
  src tilelang testing
```

确认：

- producer、cleanup 和 TODO 均已删除；
- 测试文档中允许保留历史问题描述，但不能存在实现依赖。

### 8.2 构建

优先复用当前 checkout 已配置的 `build` 目录，避免覆盖用户现有 CMake 选项：

```bash
cmake --build build -j
```

如果 `build` 尚未配置，应先检查 `CMakeCache.txt` 和当前 backend 设置，再决定使用独立 build 目录。不要未经确认重配或覆盖已有构建。

纯 C++ 开发可从仓库根目录使用：

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

TileLang dev checkout 会从 `build/lib/` 和 `build/tvm/` 加载 native library。

### 8.3 格式与静态质量

对所有修改文件运行仓库已有 formatter/linter：

- C++：clang-format 或仓库封装的 format check；
- Python：仓库配置的 formatter/linter；
- 检查新增注释只解释 representation/invariant，不重复代码；
- 检查 C++/Python 参数命名一致使用 `thread_index`；
- 检查 FFI 参数顺序与注册接口没有无意变化。

### 8.4 Pass-boundary 测试

#### LayoutInference boundary

构造 CPU target kernel，执行：

```text
BindTarget
-> MaterializeKernelLaunch(lower_thread_binding=False)
-> LayoutInference
```

断言：

- `parallel_loop_predicate` 中不含 synthetic `Var`；
- CPU predicate 是将 logical thread index 取 0 后的结果；
- `thread_bounds` 为 `[0,1)`；
- fragment layout inference 结果仍正确。

GPU 对应测试断言：

- predicate 使用真实 `threadIdx.x` identity；
- 该变量受 `thread_extent` 绑定；
- thread extent 与改动前一致。

#### LowerTileOp boundary

构造 issue #2202/#2226 的最小 while + fragment kernel，执行 CPU 前置 pass、`LayoutInference`、`LowerTileOp`，然后断言：

```python
tvm.tirx.analysis.undefined_vars(func.body, func.params)
```

不包含任何 synthetic thread variable。

补充 visitor 检查编译器生成的 fallback 不存在。不能只依赖 IR 字符串，因为字符串无法区分合法同名变量与 synthetic identity。

同时增加 name-collision regression：

- PrimFunc 有合法参数或局部变量名 `v_thread`；
- kernel 同时触发 CPU fragment/tile lowering；
- pass 后该变量引用仍存在且语义未被替换为 0。

### 8.5 ABI 测试

通过 CPU pipeline 跑到：

```text
AnnotateDeviceRegions
-> SplitHostDevice
```

断言：

- device PrimFunc params 只包含真实 buffer/scalar 参数；
- params 中没有 synthetic thread index；
- host launch 参数数量与 device function signature 一致；
- host/device PrimFunc 均不存在未定义 synthetic variable。

优先复用：

- `testing/python/transform/test_tilelang_transform_split_host_device.py` 中的 module helper；
- `CompiledArtifact.host_mod`/`device_mod`，如果该层 API 在 CPU target 下稳定。

不应把生成 C source 中不出现字符串 `v_thread` 作为唯一断言；source 检查可以作为 ABI test 的补充。

### 8.6 聚焦回归测试

每个 PR 至少运行与改动面对应的聚焦测试。最终 PR 运行：

```bash
python -m pytest \
  testing/python/transform/test_tilelang_transform_lower_tile_op.py \
  -x -vv

python -m pytest \
  testing/python/transform/test_tilelang_transform_layout_inference.py \
  -x -vv

python -m pytest \
  testing/python/transform/test_tilelang_transform_split_host_device.py \
  -x -vv

python -m pytest \
  testing/python/cpu/test_tilelang_cpu_while_repro.py \
  -x -vv

python -m pytest \
  testing/python/llvm/test_tilelang_llvm_while_repro.py \
  -x -vv
```

### 8.7 共享 API/backend 回归

因为 `LowerArgs`、`PartitionLoop` 和 `LowerParallelLoop` 是共享接口，还需要按当前环境能力运行：

```bash
python -m pytest testing/python/cpu -x -vv
python -m pytest testing/python/llvm -x -vv
python -m pytest testing/python/metal -x -vv
```

CUDA/ROCm 环境可用时，至少覆盖：

- copy lowering；
- parallel loop/layout inference；
- fill/transpose/reduce；
- GEMM/GEMM-SP FFI lowering；
- non-zero thread range min 相关用例。

当前环境不可用的 backend 应明确记录为“未运行，由 CI 覆盖”，不能把 pytest skip 当作实际通过。

### 8.8 最终验收标准

Issue #2226 可关闭需同时满足：

- `LayoutInference` 不构造 synthetic CPU thread `Var`；
- `LowerTileOp` 不构造 synthetic CPU thread `Var`；
- CPU helper 从源头接收常量 0；
- GPU helper 继续接收真实、受绑定的 `threadIdx.x`；
- `LowerTileOp` 不包含 compatibility canonicalizer；
- 不存在 name-based fallback rewrite；
- 合法同名 `v_thread` 不被改写；
- `LowerTileOp` 输出不含 synthetic free var；
- `SplitHostDevice` 后 device ABI 不含 synthetic thread 参数；
- `MakePackedAPI` 无需相关兜底；
- C/LLVM while + fragment regression通过；
- 可用 backend 的共享 helper 回归通过；
- 所有 PR 在各自合入点保持构建与行为正确。
