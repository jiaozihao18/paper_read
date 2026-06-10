# 论文阅读笔记

## 1. 基本信息

* 论文标题：GenRec: A Preference-Oriented Generative Framework for Large-Scale Recommendation
* 作者：Yanyan Zou, Junbo Qi, Lunsong Huang, Yu Li, Kewei Xu, Jiahao Gao, Binglei Zhao, Xuanhua Yang, Sulong Xu, Shengjie Li（主要来自 JD.com）
* 发表年份 / 会议或期刊：2026 / arXiv 预印本（arXiv:2604.14878）
* 研究领域：推荐系统、生成式检索（Generative Retrieval）、大规模工业推荐
* 关键词：生成式推荐、Semantic ID（SID）、Page-wise NTP、Token Merger、GRPO-SR、偏好对齐、强化学习、京东 App 部署

## 2. 一句话总结

本文提出面向用户偏好的生成式推荐框架 GenRec，在单一 decoder-only 架构中通过 Page-wise NTP 监督、非对称 Token Merger 压缩与 GRPO-SR 强化学习对齐，解决工业分页场景下的标签歧义、长序列推理成本与 reward hacking 问题，并在京东 App 一个月在线 A/B 测试中实现点击量 +9.5%、交易量 +8.7% 的提升。

## 3. 研究背景与问题

* **研究背景**：现代推荐系统多采用 retrieve-and-rank 架构；生成式检索（GR）将检索重构为条件序列生成，可直接从全库生成目标物品。TIGER、LC-Rec 等工作已验证该范式有效性，但将其扩展至大规模工业系统仍面临独特挑战。
* **现有不足**：
  1. **分页请求机制下的标签歧义**：同一用户历史 $\mathcal{H}$ 在一次分页请求中可能对应多个正样本（点击、成交等），vanilla point-wise NTP 在相同输入下需拟合多个合法输出，形成 one-to-many 歧义，梯度方差大、单物品概率质量被稀释，损害 top-$K$ 精度。
  2. **长行为序列 + 多 token SID 的推理成本**：基于 Semantic ID 的物品表示需多个 token，使用户历史部分输入序列长度约为原来的 3 倍，在线推理延迟过高。
  3. **偏好对齐与 reward hacking**：直接用稀疏点击信号做 RL 对齐效果差；密集奖励模型又可能被策略"钻空子"——生成语法合法但语义无关的 SID 组合以获得非零奖励。
* **核心问题**：如何在单一 decoder-only 架构内，同时解决工业分页场景的训练歧义、长序列推理效率与用户满意度对齐三大难题。

## 4. 核心贡献

1. 提出 **Page-Wise NTP（PW-NTP）** 监督微调策略：对整个交互页而非单个物品做监督，提供更稠密梯度信号，消除 vanilla point-wise NTP 的 one-to-many 歧义。
2. 提出 **非对称 Token Merger**：在 prefilling 阶段用线性层将三 token SID 压缩为单一向量，解码阶段保持完整 SID 序列，输入长度约减 2×，精度损失可忽略。
3. 提出 **GRPO-SR** 强化学习方法：基于 Group Relative Policy Optimization，结合 Hybrid Rewards（密集偏好模型 + 相关性门控）抑制 reward hacking，并以 NLL 正则稳定训练、锚定真实用户行为。
4. 在生成式推荐上验证 **scaling laws**，并通过京东 App 大规模部署验证：点击量 +9.5%、交易量 +8.7%；GRPO-SR 对齐版本已全量上线。

## 5. 方法概述

### 整体思路

GenRec 将检索任务统一为条件序列生成：给定用户历史行为序列 $\mathcal{H}=\{v_1,\ldots,v_n\}$（按时间排列），预测后续感兴趣物品。整体采用 **SFT（PW-NTP）→ RL（GRPO-SR）** 两阶段训练，骨干为 **Qwen2.5 decoder-only** 架构，物品以 **Semantic ID（SID）** 离散表示。

### 关键模块

**（1）Semantic ID 构建**

- 用多模态模型 **Qwen2.5-VL** 联合编码物品视觉与文本，得到连续表示。
- 在领域协同对上做微调，使 embedding 具备推荐语义。
- 用 **RQ K-means** 对残差向量迭代聚类，将每个物品 $v_i$ 映射为层次化簇索引三元组：

$$
\mathrm{SID}(v_i)=\{s_i^1, s_i^2, s_i^3\}
$$

**（2）Page-Wise NTP SFT**

- **问题分析**：vanilla point-wise NTP 在相同历史 $\mathcal{H}$ 对应 $K$ 个正样本 $\{v^{(k)}\}_{k=1}^K$ 时，需最大化 $\sum_k \log P_\theta(v^{(k)}\mid\mathcal{H})$，等价于拟合均匀混合分布，造成 **cardinality mismatch**（一次会话多个 engagement 信号被拆成孤立样本对）。
- **训练输入**：按交互强度排序的用户历史 SID 序列

$$
S_u=[\mathrm{SID}(v):v\in\mathcal{H}]_{\succ}
$$

- **训练目标**：当前页内用户交互物品（下单 $\mathcal{O}$、点击 $\mathcal{C}$、曝光 $\mathcal{E}$）按交互强度排序构成页面序列

$$
Y_{\mathrm{page}}=[\mathrm{SID}(v):v\in\mathcal{O}\cup\mathcal{C}\cup\mathcal{E}]_{\succ}
$$

- **损失函数**：对整个响应序列的标准自回归 SFT 损失

$$
\mathcal{L}_{\mathrm{SFT}}=-\sum_{t=1}^{|Y_{\mathrm{page}}|}\log P_{\theta}(y_t\mid S_u, y_{<t})
$$

- **推理**：仍采用 **point-wise beam search**，每次查询生成 beam-width 个物品，与线上生产管线兼容；训练 list-wise、推理 point-wise 的非对称设计是有意为之。

**（3）Decoder-only + Token Merger**

- 三 token SID 使用户历史部分序列长度约为原来的 3 倍。
- **Token Merger**（仅用于 prefilling）：将同一物品的三 token embedding 拼接后经线性层投影为统一向量

$$
\mathbf{h}_{v_i}=\text{Linear}(\text{Concat}(\mathbf{e}(s_i^1), \mathbf{e}(s_i^2), \mathbf{e}(s_i^3)))
$$

- 特殊 token（如 `<sep>`）保持不合并，作为结构分隔符。
- 解码阶段与生成目标仍使用原始完整 SID token 序列；prompt 长度约减 **2×**。

**（4）GRPO-SR 偏好对齐**

- RL 阶段与线上一致：每个 rollout 每次查询生成**单个**物品序列（point-wise）。
- 优化组内**相对偏好**而非绝对奖励值，提升工业场景鲁棒性。

*Hybrid Reward 设计*：

- 用 **SIM 模型**估计连续偏好分 $r_i^{\mathrm{pref}}\in[0,1]$（原始点击信号过于稀疏）。
- **相关性门控**抑制 reward hacking：$\mathcal{G}_i=\mathbb{I}(s_i>\tau)$（$\tau$ 为小常数，$s_i$ 为相关性分数）

$$
r_i=\mathcal{G}_i\cdot r_i^{\mathrm{pref}}
$$

- **正样本锚定校准**：令 $\mathcal{D}^+=\mathcal{O}\cup\mathcal{C}$，将真实交互物品奖励锚定到组内最大值

$$
\tilde{r}_i=\bigl[1-\mathbb{I}(o_i\in\mathcal{D}^+)\bigr]\cdot r_i+\mathbb{I}(o_i\in\mathcal{D}^+)\cdot r_{\max}
$$

*GRPO-SR 目标*：

$$
\mathcal{L}_{\mathrm{GRPO-SR}}(\theta)=-\mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\frac{\pi_{\theta}(o_{i,t}\mid\cdot)}{\mathrm{sg}(\pi_{\theta}(o_{i,t}\mid\cdot))}\hat{A}_{i,t}\Bigg]-\alpha\cdot\mathbb{E}_{v\sim\mathcal{D}^+}\left[\sum_{t=1}^{|v|}\log\pi_{\theta}(v_t\mid\cdot)\right]
$$

- 第一项：重要性采样 $\pi_\theta/\mathrm{sg}(\pi_\theta)$ 实现稳定的一步策略更新。
- 第二项：权重 $\alpha$ 的 **NLL 正则**，在正样本轨迹 $\mathcal{D}^+$ 上约束策略，区别于标准 KL 惩罚，显式锚定真实用户行为。

### 与已有方法的区别

| 维度 | TIGER / LC-Rec（vanilla NTP） | GenRec |
|------|------------------------------|--------|
| 训练目标 | point-wise 单物品预测 | page-wise 整页序列监督 |
| 训练/推理 | 同构 point-wise | 训练 list-wise、推理 point-wise（有意解耦） |
| 输入效率 | 完整三 token SID | prefilling 阶段 Token Merger 压缩 |
| 偏好对齐 | 仅 SFT | SFT + GRPO-SR（Hybrid Reward + NLL 正则） |
| 工业验证 | 主要为离线实验 | 京东 App 全量部署 + 在线 A/B |

## 6. 实验设计

### 数据集

- 来源：**京东（JD.com）** 大规模推荐平台
- 规模：约 **5.6 亿**条用户交互序列，时间跨度一个月
- 划分：最后一天为测试集，其余为训练集

### 对比方法 / baseline

| 类型 | 方法 |
|------|------|
| 传统序列推荐 | BERT4Rec、SASRec |
| 生成式推荐 | TIGER、LC-Rec（均用 vanilla point-wise NTP 训练） |

- LC-Rec 用相同 Qwen2.5 变体复现，作为公平对比。
- GenRec 消融：**w/o TM**（移除 Token Merger）、**w/o $\mathcal{G}$**（移除相关性门控）等。

### 评价指标

| 阶段 | 指标 |
|------|------|
| SFT | **HR@K**（HitRate）、**N@K**（NDCG）、**HaR**（Hallucination Rate，无效 SID 占比，↓） |
| RL | **R@K**（Reward Metrics）：生成 $K$ 个物品中最高 $r^{\mathrm{SIM}}$ 分数 |
| 在线 | 曝光率、点击量、交易量（双侧检验 $p<0.05$） |

### 实验设置

- 骨干：**Qwen2.5** decoder-only，规模 1.5B / 3B / 7B（主实验用 3B）
- 硬件：8× NVIDIA H100，分布式训练
- 优化器：**AdamW**；前 1% 步数线性 warm-up，之后 cosine 学习率衰减

### 消融与额外分析

- Token Merger 有效性（GenRec vs w/o TM）
- PW-NTP 有效性（GenRec vs LC-Rec / vanilla NTP）
- 模型规模 scaling（1.5B → 3B → 7B）
- RL 方法对比（GRPO、GRPO-SR、有无门控 $\mathcal{G}$）
- 在线 A/B：基础 SFT 与 GRPO-SR 对齐版本各 10% 流量，持续一个月
- 长尾物品专项分析

## 7. 主要实验结果

### 离线 SFT 性能（Table 1，Qwen2.5 3B）

GenRec 在 HR、NDCG 上全面优于传统与生成式基线，HaR 显著更低：

| 方法 | HR@1 | HR@50 | N@50 | HaR |
|------|------|-------|------|-----|
| BERT4Rec | 0.0315 | 0.1832 | 0.0689 | - |
| SASRec | 0.0383 | 0.1976 | 0.0776 | - |
| TIGER | 0.0518 | 0.3556 | 0.1409 | 15.46% |
| LC-Rec | 0.0947 | 0.6226 | 0.2717 | 7.80% |
| **GenRec** | **0.1189** | **0.7192** | **0.3247** | **4.96%** |
| w/o TM | 0.1193 | 0.7201 | 0.3276 | 4.89% |

- GenRec 相对 LC-Rec 在 HR@1 上提升约 **25.6%**（0.1189 vs 0.0947）。
- **w/o TM 与完整 GenRec 性能相当**，说明 Token Merger 在约 2× 压缩输入的同时几乎不损失精度。
- PW-NTP 使 **HaR 降低超过 50%**（相对 LC-Rec 的 7.80%），联合预测促进更连贯的物品生成。

### 模型 Scaling（Table 2）

| 规模 | HR@50 | N@50 | HaR |
|------|-------|------|-----|
| 1.5B | 0.6527 | 0.1885 | 5.34% |
| 3B | 0.7192 | 0.3247 | 4.96% |
| 7B | 0.7216 | 0.3269 | 5.42% |

- 训练 loss 随规模单调下降，但 **3B→7B 增益远小于 1.5B→3B**（参数量增约 2.3×）。
- 作者分析：3B 更深更窄（36 层 / 2048 hidden）vs 7B 更浅更宽（28 层 / 3584 hidden），在生成式推荐中**深度可能比宽度更重要**（"capacity density" 假说）。

### RL 对齐（Table 3）

| 方法 | HR@50 | R@1 | R@50 | HaR |
|------|-------|-----|------|-----|
| Base SFT | 0.7192 | 0.1027 | 0.1776 | 4.96% |
| GRPO | 0.7248 | 0.1177 | 0.1861 | 6.03% |
| **GRPO-SR** | **0.7438** | **0.1212** | **0.1892** | **2.68%** |
| GRPO-SR w/o $\mathcal{G}$ | 0.7016 | 0.1067 | 0.1813 | 1.96% |

- GRPO-SR 在所有推理预算下均优于 SFT 基线；**R@1 相对提升 +18.01%**。
- 移除门控 $\mathcal{G}$ 后：奖励略有上升，但 **HR@50 大幅下降、HaR 异常偏低**——典型 reward hacking（利用 SIM 对合法 SID 的要求，牺牲整体质量）。
- GRPO-SR 在提升偏好对齐的同时将 HaR 降至 **2.68%**，兼顾质量与有效性。

### 在线 A/B 测试（Table 4，京东首页 Feed，各 10% 流量，一个月）

| 设置 | 曝光率 | 点击量 | 交易量 |
|------|--------|--------|--------|
| Base SFT | 48.7% | +8.5% | +7.3% |
| + GRPO-SR | 57.3% | **+9.5%** | **+8.7%** |

- 长尾物品：曝光率 +10%、点击量 +16%、交易量 +13%。
- **GRPO-SR 对齐版本已全量部署生产**。

### 作者对结果的解释

1. PW-NTP 通过聚合页面级监督，解决 one-to-many 歧义并提供更稠密学习信号，加速收敛、降低幻觉率。
2. Token Merger 在 prefilling 阶段保留关键信息的同时满足严格推理预算。
3. GRPO-SR 的组相对优化 + NLL 锚定 + 相关性门控，有效将输出分布推向高奖励候选并抑制 hacking。
4. Scaling 分析表明生成式推荐中存在 scaling laws，但收益随规模递减，架构形状（深 vs 宽）影响显著。

## 8. 论文优点

* **问题定位精准**：直面工业分页、长序列、偏好对齐三大真实痛点，而非仅在学术数据集上验证生成式检索。
* **训练/推理解耦设计合理**：PW-NTP 训练 + point-wise beam search 推理，兼顾监督信号密度与线上管线兼容性。
* **Token Merger 简洁高效**：非对称压缩（prefill 合并、decode 完整）是低侵入性的工程优化，消融证明几乎零精度损失。
* **GRPO-SR 设计周全**：Hybrid Reward、正样本锚定、NLL 正则、相关性门控形成完整防护链，且有移除门控的反面证据。
* **实验链条完整**：离线对比 + scaling + RL 消融 + 一个月在线 A/B + 全量部署，工业可信度极高。
* **与 LLM 生态对齐**：直接采用 Qwen2.5 decoder-only，可复用 LLM 推理优化技术栈。

## 9. 局限性与不足

* **作者自述**：未来将探索该框架的**推理能力**（reasoning ability）；当前未深入展开。
* **数据集单一**：实验数据全部来自京东，跨平台、跨品类泛化性未验证。
* **作者信息在提取文本中缺失**：`output_GenRec.txt` 由 HTML 提取时未包含作者与发表 venue 元数据（需从 HTML 补充）。
* **表格细节受限**：部分 baseline（BERT4Rec、SASRec）无 HaR 数据；K 的具体取值原文未在提取文本中明确说明。
* **门控阈值 $\tau$ 与 SIM 细节**：$\tau$ 选取、$s_i$ 的具体定义与 SIM 模型训练细节原文未充分展开。
* **RL 计算成本**：GRPO 需多 rollout 采样，训练成本高于纯 SFT，文中未量化额外开销。
* **PW-NTP 目标序列含曝光物品 $\mathcal{E}$**：将曝光纳入监督是否合理、对噪声的鲁棒性讨论有限。
* **可解释性**：生成式检索的黑盒特性仍存，SID 语义与推荐决策的可解释性未涉及。

## 10. 可借鉴之处

1. **分页场景下的 list-wise 训练 + point-wise 推理**范式，可直接迁移到其他存在"同输入多标签"的工业推荐场景。
2. **非对称 Token Merger**：对多 token 物品表示的 prefilling 压缩思路，适用于任何 SID/多模态 token 序列过长的生成式推荐系统。
3. **GRPO-SR 三板斧**：组相对奖励 + NLL 行为锚定 + 相关性门控，为 RLHF/RLAIF 在推荐场景的 reward hacking 提供了可复现的防护模板。
4. **正样本奖励锚定**（命中 $\mathcal{D}^+$ 的候选强制获得组内最高奖励），简单有效地缓解奖励模型低估真实交互物品的问题。
5. **HaR 作为生成式推荐专属指标**，与 HR/NDCG 互补，衡量 SID 合法性，值得在同类工作中采用。
6. **工业论文评估规范**：离线 + scaling + 消融 + 长周期在线 A/B + 全量部署的完整证据链，是工业推荐论文的标杆写法。

## 11. 可延伸的研究方向

1. **跨平台泛化与领域自适应**：将 PW-NTP + Token Merger + GRPO-SR 迁移至视频、本地生活等其他分页推荐场景，研究 SID 码本与奖励模型的领域迁移。
2. **推理增强的生成式推荐**：按作者展望，探索在 GenRec 框架中引入显式推理链（用户意图分析 → 候选生成），提升可解释性与复杂意图建模。
3. **更高效的 RL 对齐**：研究单步/离线 RL、奖励模型蒸馏等方案，降低 GRPO 多 rollout 的训练成本，支撑日更甚至实时对齐。
4. **曝光噪声鲁棒性**：分析 PW-NTP 中 $\mathcal{E}$（曝光未点击）的监督权重与筛选策略，减少负向噪声对生成的干扰。
5. **架构 Scaling 深入分析**：在生成式推荐中系统比较深窄 vs 浅宽 Transformer、MoE 等结构，验证"capacity density"假说并指导工业模型选型。
