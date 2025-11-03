# 工业级A卡（申请评分卡）风控模型

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 1. 项目概述

本项目包含了一个完整的、工业级的机器学习流程，用于构建一个A卡（Application Scorecard）消费者信用风险模型。模型基于 [LendingClub 公开数据集](https://www.lendingclub.com/info/download-data.action) (2007-2018年)，旨在预测贷款申请人在未来是否会发生违约。

本项目的核心是**对比三种主流 GBDT 算法 (XGBoost, LightGBM, CatBoost)**，并采用严格的**时间外 (Out-of-Time, OOT)** 验证策略来选拔出一个兼具**高区分度 (AUC/KS)** 和**高稳定性 (PSI)** 的冠军模型。

---

## 2. 核心技术与特色

本项目区别于标准的数据科学竞赛项目，严格遵循了金融风控行业的最佳实践：

* **严格OOT验证：** 杜绝数据穿越。使用 2012-2013H1 数据进行训练，2013H2 数据进行验证（调参），2014 年全年数据作为最终的 OOT 测试集。
* **三大模型横评：** 完整实现了 XGBoost, LightGBM, 和 CatBoost 的训练-评估流程。
* **业务逻辑约束：** 在所有模型训练中均加入了**单调性约束** (Monotonicity Constraints)，确保模型行为符合金融常识（例如：FICO 分数越高，风险越低）。
* **自动化超参调优：** 使用 `Optuna` 在 OOT 验证集上对三个模型进行高效的超参数搜索。
* **稳定性监控 (PSI)：** 严格计算模型分 PSI 和所有核心特征的 PSI，确保模型在部署后不会因客群漂移而迅速失效。
* **工业级评分卡转换：**
    1.  **概率校准：** 对因样本不均衡（`scale_pos_weight`）而失真的模型概率进行校准。
    2.  **两点锚定法：** 采用比传统PDO法更直观的“两点锚定法”将概率转换为业务评分（例如：锚定 5% 违约率=800分，20% 违约率=600分）。

---

## 3. 数据与建模流程

1.  **数据清洗与预处理:**
    * 加载 `accepted_2007_to_2018q4.csv` 数据。
    * **时间筛选:** 保留 2007-01-01 至 2014-12-31 的数据用于模型构建。
    * **目标变量 (Y) 定义:** `loan_status` 为 'Charged Off' 或 'Default' 定义为 1 (坏客户)，'Fully Paid' 定义为 0 (好客户)。
    * **特征剔除:** 移除所有贷后特征、未来信息特征、以及高无关/高缺失特征。

2.  **特征工程:**
    * 对核心原始变量（如 `emp_length`）进行格式转换。
    * 构建核心衍生变量，如 `credit_history_months` (信用历史月数)。
    * 筛选出用于建模的核心稳定特征（约15-20个）。
    * 构建稳健的比率型特征，如 `loan_to_income_ratio` (贷收比)。

3.  **OOT 验证策略 (滚动时间窗口):**
    * **训练集 (Train):** `2012-01-01` -> `2013-06-30`
    * **验证集 (Validation):** `2013-07-01` -> `2013-12-31` (用于 Optuna 调参和模型早停)
    * **测试集 (Test-OOT):** `2014-01-01` -> `2014-12-31` (用于最终的模型选型和报告)

4.  **模型选型 (OOT 测试集对比):**
    * 三个模型均在 (训练集) 上训练，在 (验证集) 上调参。
    * 最终在 (测试集-OOT) 上的表现对比如下：

| 评估指标 | XGBoost | LightGBM | **CatBoost (冠军)** | 工业界标准 |
| :--- | :--- | :--- | :--- | :--- |
| **区分度 (AUC)** | 0.7039 | 0.7143 | **0.7160** | > 0.70 (良好) |
| **区分度 (KS)** | 0.2964 | 0.3129 | **0.3162** | > 0.25 (良好) |
| **模型分 PSI** | 0.0067 | 0.0034 | 0.0066 | < 0.10 (优秀) |

**结论：** CatBoost 在 OOT 测试集上的 AUC 和 KS 表现均微弱领先，且保持了极高的稳定性，被选为**冠军模型 (Champion Model)**。

---

## 4. 冠军模型 (CatBoost) 深度分析

冠军模型 (CatBoost) 在性能、稳定性和业务可用性上均表现出色。

### 4.1. 风险区分度 (AUC & KS)

* **ROC 曲线 (OOT):** AUC 达到 **0.7160**，模型整体排序能力强。
* **KS 曲线 (OOT):** KS 达到 **31.62%**，在 `Score=637` 时取得最大分离度，区分能力优秀。

<img width="696" height="542" alt="image" src="https://github.com/user-attachments/assets/dcb10e88-07aa-4ab0-a3b8-fc22dd8ab1ab" />
<img width="850" height="541" alt="image" src="https://github.com/user-attachments/assets/d07be8da-0db8-4cf8-947b-63bcb3197d09" />



### 4.2. 业务可用性 (分数分布)

* 采用“两点锚定法”后，分数分布直观，业务逻辑清晰。
* 好客户 (Y=0) 均分为 **678**，坏客户 (Y=1) 均分为 **602**，平均分差 **76.2分**，为策略部署提供了充足空间。

<img width="857" height="541" alt="image" src="https://github.com/user-attachments/assets/a3cb3dc6-c5a3-4987-a728-a25fbac8bd0c" />


### 4.3. 模型稳定性 (PSI)

* 模型分 PSI 仅为 **0.0066**，远低于 0.1 的行业警戒线。
* 下图可见，训练集 (蓝色) 和 OOT 测试集 (橙色) 的分数分布**几乎完全重合**，表明模型极其稳定，抗客群漂移能力强。

<img width="865" height="541" alt="image" src="https://github.com/user-attachments/assets/94ee29e5-f71f-4e10-8131-e79ad0586b66" />


---

## 5. 如何运行

1.  **克隆仓库:**
    ```bash
    git clone [https://github.com/](https://github.com/)[Your-Username]/[Your-Repo-Name].git
    cd [Your-Repo-Name]
    ```

2.  **创建虚拟环境 (推荐):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # (On Windows: venv\Scripts\activate)
    ```

3.  **安装依赖:**
    (请根据您的 `requirements.txt` 文件进行调整)
    ```bash
    pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna matplotlib seaborn
    ```
    *建议您运行 `pip freeze > requirements.txt` 来生成依赖文件。*

4.  **准备数据:**
    * 下载 `accepted_2007_to_2018q4.csv` 文件。
    * 将其放置在与 Python 脚本相同的根目录中。

5.  **运行模型:**
    * 您可以选择运行任一模型的 pipeline 脚本（例如 `catboost_pipeline.py`）。
    ```bash
    python catboost_pipeline.py
    ```

---


## 6. 许可 (License)

本项目采用 MIT 许可。
