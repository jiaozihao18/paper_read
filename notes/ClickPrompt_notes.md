# 论文阅读笔记

## 1. 基本信息

* 论文标题：ClickPrompt: CTR Models are Strong Prompt Generators for Adapting Language Models to CTR Prediction
* 作者：Jianghao Lin, Bo Chen, Hangyu Wang, Yunjia Xi, Yanru Qu, Xinyi Dai, Kangning Zhang, Ruiming Tang, Yong Yu, Weinan Zhang（上海交通大学、华为诺亚方舟实验室）
* 发表年份 / 会议或期刊：2024（原文未明确说明具体会议或期刊）
* 研究领域：推荐系统、点击率（CTR）预测、预训练语言模型（PLM）适配
* 关键词：Pretrained Language Models, CTR Prediction, Soft Prompt, PA-MLM, 协同知识对齐

## 2. 一句话总结

本文提出模型无关框架 ClickPrompt，将 CTR 模型作为 PLM 的 soft prompt 生成器，通过 PA-MLM 预训练在 prompt 接口上显式对齐协同知识与语义知识；下游既可联合微调 PLM 取得最优效果，也可仅微调 CTR 模型以零额外推理开销获得语义感知的参数初始化。

## 3. 研究背景与问题

* **研究背景**：CTR 预测是在线广告、推荐等互联网应用的核心组件，输入通常为多域类别型（multi-field categorical）数据。传统 CTR 模型将特征 one-hot 编码为 ID 特征，擅长挖掘特征交互等协同信号，但会丢失语义信息（如电影名、职业、颜色等文本含义），在冷启动用户/物品、长尾低频特征、点击信号不足等场景下表现受限。
* **现有不足**：
  - **传统 ID-based CTR 模型**：one-hot 编码导致语义信息损失，无法利用特征间的隐式语义关联。
  - **PLM-based CTR 方法**（如 CTR-BERT、P5、PTab）：通过 hard prompt 模板将数据文本化，保留语义，但存在两大局限：
    1. **预测不准**：难以建模协同知识——纯 ID 特征（用户 ID、物品 ID）对 PLM 无意义；字段级特征交互在文本线性拼接后被 token 化，丢失 field-level 交互视角。
    2. **推理低效**：PLM 参数量大、Transformer 层数深，在线服务需在数十毫秒内响应，直接部署 PLM 不可接受；预缓存 PLM 表征虽可加速，但消耗存储与工程成本，且损害实时性。
* **核心问题**：如何同时建模语义知识与协同知识以实现准确 CTR 估计，并解决 PLM 带来的推理效率问题。

## 4. 核心贡献

1. 提出 ClickPrompt 框架，将 CTR 模型视为 PLM 的 soft prompt 生成器，设计 PA-MLM（Prompt-augmented Masked Language Modeling）预训练任务，通过 soft prompt 接口实现协同知识与语义知识的显式交互与对齐。
2. ClickPrompt 模型无关，兼容多种 CTR 模型与 PLM；提供两种下游微调策略——联合微调 PLM 取得最优性能，或仅微调 CTR 模型在不改变结构、不增加推理开销的前提下提升精度。
3. 在 MovieLens-1M、BookCrossing、Amazon-Toys、GoodReads 四个真实公开数据集上验证有效性，全面优于现有 baseline。

## 5. 方法概述

### 整体思路

ClickPrompt 维护一个 CTR 模型和一个 PLM，分别接收 ID 特征 $x_i^{ID}$ 和文本特征 $x_i^{text}$。CTR 模型上方增设 prompt generation layer，生成可学习的 soft prompt 向量，作为 prefix hidden states 注入 PLM 每一层 Transformer。采用 pretrain-finetune 范式：

1. **预训练**：PA-MLM 任务，PLM 需基于文本上下文 + CTR 模型生成的 soft prompt 恢复被 mask 的 token，使协同知识经前向传播流入 PLM，语义知识经反向传播回流 CTR 模型。
2. **微调**：可选「with PLM」或「w/o PLM」两种策略。

### 关键模块

**（1）模态转换（Modality Transformation）**

- ID 特征：标准 one-hot 编码。
- 文本特征：采用简洁的 "what is what" hard prompt 模板（避免复杂模板误导模型）：

$$
x_{i,j}^{text} = [f_j^{name},\ \text{``is''},\ f_{i,j},\ \text{``.''}],\quad j=1,\ldots,F
$$

$$
x_i^{text} = [x_{i,1}^{text},\ x_{i,2}^{text},\ \cdots,\ x_{i,F}^{text}]
$$

其中 $f_j^{name}$ 为第 $j$ 个字段名，$f_{i,j}$ 为第 $i$ 个样本该字段的取值。

**（2）Prompt 生成（Prompt Generation）**

CTR 模型的 embedding layer 与 feature interaction (FI) layer 产生紧凑表示 $q_i$：

$$
q_i = \text{FI\_Layer}(\text{Embed\_Layer}(x_i^{ID}))
$$

通过 $L \times K$ 组并行投影网络 $g_{l,k}(\cdot)$（tanh 激活的两层 MLP）生成每层、每位置的 soft prompt：

$$
p_{i,l,k} = g_{l,k}(q_i),\quad 1 \le l \le L,\ 1 \le k \le K
$$

其中 $L$ 为 PLM Transformer 层数，$K$ 为每层 prompt 数量。

**（3）Prompt 融合（Prompt Fusion）**

文本 token 化后得到 $Z$ 个 word token，第 $l$ 层 Transformer 计算为：

$$
[h_{i,l+1,z}]_{z=1}^{Z} = \text{Transformer}_l\left([p_{i,l,k}]_{k=1}^{K} \oplus [h_{i,l,z}]_{z=1}^{Z}\right)
$$

soft prompt 作为 prefix hidden states，经 self-attention 与文本语义显式对齐融合。最终经 Pooling + MLP 输出预测。

**（4）PA-MLM 预训练**

对文本特征 $x_i^{text}$ 做 token masking（15% token，8:1:1 比例分别替换为 [MASK]、随机词、保持不变），ID 特征保持完整。PLM 需结合文本上下文与 soft prompt 恢复被 mask 的 token，使用 softmax + 交叉熵损失。Pooling & prediction 层此时充当语言模型 decoder。

**（5）微调策略**

*Finetune with PLM*：保留完整结构，CTR 与 PLM 预测加权融合：

$$
\hat{y}_i^{CTR} = \text{MLP}(q_i)
$$

$$
\hat{y}_i^{PLM} = \text{MLP}\left(\text{Pooling}\left([h_{i,L+1,z}]_{z=1}^{Z}\right)\right)
$$

$$
\hat{y}_i = \sigma\left(\alpha \times \hat{y}_i^{CTR} + (1-\alpha) \times \hat{y}_i^{PLM}\right)
$$

$\alpha$ 为可学习权重，$\sigma$ 为 sigmoid。

*Finetune w/o PLM*：仅微调 CTR 模型，PA-MLM 预训练已将语义知识注入 CTR 参数：

$$
\hat{y}_i = \sigma\left(\text{MLP}(q_i)\right)
$$

两种策略均使用 BCE 损失：

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]
$$

### 与已有方法的区别

| 维度 | 传统 CTR | 纯 PLM CTR | CTRL | ClickPrompt |
|------|---------|-----------|------|-------------|
| 语义信息 | 丢失 | 充分利用 | 对比学习蒸馏 | soft prompt 显式对齐 |
| 协同/字段交互 | 核心建模 | 难以捕获 | 保留协同主干 | CTR 模型生成 prompt |
| 知识对齐粒度 | — | — | 实例级粗粒度对比 | 逐层 prompt 细粒度对齐 |
| 推理效率 | 高 | 低 | 仅部署 CTR | 可选 w/o PLM，零额外开销 |

与 CTRL 的关键区别：CTRL 通过 CLIP 式对比学习在最终表征层做实例级隐式对齐；ClickPrompt 通过 layerwise soft prompt 在 PLM 每一层实现显式 early interaction 与细粒度对齐。

## 6. 实验设计

* **数据集**：MovieLens-1M、BookCrossing、Amazon-Toys、GoodReads（均需保留原始语义/文本特征，非匿名 ID）。按全局时间戳 8:1:1 划分训练/验证/测试集。评分二值化阈值：ML-1M 和 AZ-Toys 为 4（剔除评分为 3 的中性样本），BookCrossing 为 5，GoodReads 为 4；AZ-Toys 5-core 过滤，GoodReads 10-core 过滤。
* **对比方法 / baseline**：
  - 传统 CTR：FM, DNN, DeepFM, xDeepFM, PNN, DCN, AutoInt, FiGNN, FGCNN, DCNv2
  - PLM-based：CTR-BERT, P5, PTab, CTRL
* **评价指标**：AUC（越高越好）、Log Loss（越低越好）；CTR 领域 0.001 量级的提升即可视为显著。
* **实验设置**：
  - 默认骨干：DCNv2（CTR）+ RoBERTa-base（PLM，125M）
  - 优化器：AdamW
  - PA-MLM 预训练：batch size 1024，lr $5 \times 10^{-5}$，warm-up ratio $\{0, 0.05, 0.1\}$，20 epoch
  - 微调：CTR 部分 lr $1 \times 10^{-3}$，PLM 部分 lr $\{0, 3 \times 10^{-5}, 5 \times 10^{-5}\}$（lr=0 即冻结 PLM）；batch size 因数据集而异（256/1024/4096）
  - 每层 prompt 数 $K \in \{1, 3, 5, 7\}$；投影网络 hidden size 等于 PLM embedding size
  - 选取验证集 AUC 最高的 checkpoint 在测试集评估
* **消融实验与额外分析**：
  - Prompt 策略：layerwise vs 仅输入层 prompt
  - 融合策略消融：w/o Prompt、w/o Pretrain、w/o Both
  - 模型兼容性：不同 CTR 骨干（DCNv2, AutoInt, DNN）× 不同 PLM（TinyBERT 14.5M, RoBERTa-base 125M, RoBERTa-large 335M）
  - 长尾用户/物品分析：训练集频次 bottom 10% 为长尾
  - 附录：Case Study（attention 权重）、推理时间对比、GPT2/BART 架构兼容性（GPT2 用 PA-CLM）

## 7. 主要实验结果

* **整体性能（RQ1）**：
  - 传统 CTR 模型整体显著优于 PLM-based 方法（CTRL 除外），说明协同信息对 CTR 至关重要，仅靠语义输入效果不佳。
  - CTRL 在 baseline 中表现最好（CLIP 框架 + 对比预训练蒸馏语义知识），但对比目标仅提供实例级粗粒度监督，不如 ClickPrompt 的显式 early interaction。
  - **ClickPrompt with PLM** 全面显著优于所有 baseline，验证 soft prompt 接口上显式对齐的有效性。
  - **ClickPrompt w/o PLM** 通常排名第二，在不改变 DCNv2 结构的前提下显著超越 baseline，证明 PA-MLM 语义感知初始化的价值。
* **模型兼容性（RQ2）**：
  - 所有 CTR 骨干和 PLM 骨干上，ClickPrompt 均显著优于原始 CTR 模型（N/A）。
  - PLM 越大，性能提升越明显（开放世界知识更丰富），但增益不成比例；综合考虑训练开销，RoBERTa-base 是性价比最优选择。
* **消融实验（RQ3）**：
  - Layerwise prompt 一致优于仅在输入层放置 prompt（浅层 prompt 易被 PLM 前向传播淹没）。
  - 移除 prompt 接口或 PA-MLM 预训练均导致三个数据集、三个 CTR 骨干上的性能下降；两者缺一不可。
* **长尾分析（RQ4）**：
  - DCNv2 在长尾用户/物品上性能显著下降；ClickPrompt 在所有四个子集（长尾/非长尾用户 × 长尾/非长尾物品）上均有一致提升。
  - 长尾问题越严重（用户和物品均为长尾），ClickPrompt 相对 DCNv2 的提升越大，说明语义知识对冷启动/长尾场景尤为有效。
* **附录关键发现**：
  - Case Study：PLM 对不同字段 mask 重建时，会对同一组 soft prompt 自适应分配不同 attention 权重，甚至将部分 prompt 权重置近零以过滤无关信息。
  - 推理时间：ClickPrompt with PLM 性能最优但推理成本较高；w/o PLM 策略在不增加推理开销的前提下提升 AUC。

## 8. 论文优点

* **创新性**：首次将 CTR 模型定位为 PLM 的 soft prompt 生成器，用 layerwise prompt 桥接两种模态，比对比学习（CTRL）更细粒度的 early interaction 设计新颖。
* **方法设计**：模型无关框架，架构清晰（模态转换 → prompt 生成 → prompt 融合），pretrain-finetune 两阶段分工明确；w/o PLM 策略巧妙解决推理效率问题。
* **实验充分性**：四个数据集、14 个 baseline、兼容性实验、消融、长尾分析、推理时间、不同 PLM 架构（encoder-only/decoder-only/encoder-decoder）验证，覆盖面广。
* **写作清晰度**：问题动机（语义损失 vs 协同缺失 vs 推理低效）层次分明，与 CTRL 等工作的对比定位清楚。

## 9. 局限性与不足

* **作者自述的 limitation**（结论部分）：
  - 预训练效率仍有提升空间。
  - 尚未探索在其他推荐任务（如 learning to rank）上的应用。
* **方法假设**：
  - 依赖数据集保留原始语义/文本特征，匿名 ID 的工业数据无法直接使用。
  - PA-MLM 预训练 + 联合微调 PLM 的训练成本仍然较高。
* **实验局限**：
  - 原文未报告工业在线 A/B 测试（CTRL 有华为工业验证）。
  - w/o PLM 策略虽零推理开销，但性能略低于 with PLM，存在精度-效率权衡。

## 10. 可借鉴之处

* **双模态融合范式**：不强行将 ID 特征文本化或丢弃协同信号，而是保留两条独立通路，通过 soft prompt 作为接口对齐——对任何需要融合结构化 ID 特征与自然语言语义的推荐/广告场景均有参考价值。
* **预训练任务设计**：PA-MLM 迫使 PLM 从 prompt 中提取协同信息以完成 mask 恢复，是比纯对比学习更细粒度的对齐监督，可借鉴到表格-文本跨模态预训练。
* **推理友好的两阶段策略**：预训练阶段充分利用大模型，推理阶段仅部署轻量 CTR 模型——与 CTRL 思路一致，是工业落地的实用范式。
* **简洁 prompt 模板**："field name is value." 的简单模板优于复杂模板，对表格数据文本化有指导意义。
* **长尾/冷启动分析**：按频次划分测试子集评估的方法可复用于评估语义增强方案的实际收益。

## 11. 可延伸的研究方向

1. **提升预训练效率**：作者明确提出；可探索蒸馏、参数高效微调（LoRA 等）或更轻量的 prompt 生成器以减少 PA-MLM 训练开销。
2. **扩展至其他推荐任务**：learning to rank、序列推荐、多任务推荐等，验证 soft prompt 桥接机制的泛化性。
3. **工业匿名数据适配**：在无法获取原始文本特征的场景下，如何用 LLM 生成伪文本或外部知识增强 prompt 生成。
4. **与更大规模 LLM 结合**：探索 decoder-only LLM（如 GPT 系列）在 PA-CLM 框架下的 scaling law 与工业可行性。
5. **动态/自适应 prompt 策略**：Case Study 显示 PLM 会自适应调整 prompt attention 权重，可进一步研究可学习的 prompt 数量选择或字段感知的 prompt 路由机制。
