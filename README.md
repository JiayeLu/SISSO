# SISSO: Fixed Descriptors & Per-Dimension Feature Constraints

本分支在 SISSO 3.5 的基础上新增了两项可选功能，便于在多维描述符搜索中精确控制特征来源：

1) 固定描述符（fix_descriptor）
2) 按描述符维度限制主特征采样（constrain_pf）

## 1. 固定描述符

- 作用：强制某些表达式出现在指定的描述符维度，模型搜索时不会被淘汰或换位。
- 配置：
fix_desc_dims=(1,3) ! 要固定的维度序号（从 1 开始）
fix_desc_exprs=(ENH3, (DI)^-1) ! 对应维度的表达式字符串，需与 Uspace.expressions 中一致


- 运行期行为：
- DI 会先在 `SIS_subspaces/Uspace.expressions` 中定位这些表达式的特征 ID。
- 在 L0 组合时，固定特征被强制加入活跃集，并在输出前按指定维度重排。
- 若固定表达式不符合当前维度的特征约束（见下文），会直接报错并终止。

## 2. 按维度限制主特征采样（constrain_pf）

- 作用：为每个描述符维度限定可使用的原始特征列（train.dat 的列号）。
- 配置：
constrain_pf_dims=(1,2,3) ! 需要约束的维度列表
constrain_pf_ranges=(5 7)(9 11)(2 4) ! 与上面维度一一对应的闭区间 [lo, hi]


- 每个区间只能有两个数字（lo hi），空格或逗号分隔均可。
- 列号按 train.dat 中的主特征顺序编号；回归时从第 3 列起为原始特征。
- 运行期行为：
- DI 读取 Uspace.expressions 后，为每个维度构建“允许特征”表，不删除原有特征 ID。
- 在模型组合时，对第 k 维的候选特征先重排固定描述符，再检查是否在该维度允许集合内；不符合直接跳过组合。
- 若某维度的约束把可用特征筛空，会报错。

## 使用示例

3 维回归，固定第一维为 ENH3，第二维只用列 9–11，第三维只用列 5–7：
desc_dim=3
fix_desc_dims=(1)
fix_desc_exprs=(ENH3)
constrain_pf_dims=(2,3)
constrain_pf_ranges=(9 11)(5 7)



## 运行与输出检查

- 设置 `restart=0` 重新开始，清理旧的 `CONTINUE/Models/SIS_subspaces`。
- `SISSO.out` 会打印：
  - Fixed descriptors 列表
  - 每个维度的 Primary feature constraints
- 若固定描述符未找到或违反约束，会在 DI 阶段报错退出。
- `Models/*/toprank*` 中的特征 ID 按最终维度顺序排列，固定维度的特征应与配置一致，每个维度的特征来源应落在对应区间生成的
