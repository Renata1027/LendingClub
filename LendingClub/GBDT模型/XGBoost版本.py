import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split # (保留，但仅用于示例)
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, ConfusionMatrixDisplay, RocCurveDisplay
from scipy.stats import ks_2samp
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
print("--- 库导入和环境设置完成 ---")
print("\n--- [阶段一] 开始数据加载与预处理 ---")
try:
    file_path = 'accepted_2007_to_2018q4.csv'
    df = pd.read_csv(file_path, low_memory=False)
    print(f"数据加载成功，原始数据形状: {df.shape}")
    df['issue_d'] = pd.to_datetime(df['issue_d'], errors='coerce')
    start_date = pd.to_datetime('2007-01-01')
    end_date = pd.to_datetime('2014-12-31')
    df_filtered = df[(df['issue_d'] >= start_date) & (df['issue_d'] <= end_date)].copy()
    print(f"筛选2007-2014年数据后，形状为: {df_filtered.shape}")
    cols_to_drop = [
        'id', 'member_id', 'url', 'desc', 'title', 'emp_title', 'pymnt_plan', 'out_prncp',
        'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
        'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d',
        'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d', 'acc_now_delinq',
        'chargeoff_within_12_mths', 'delinq_amnt', 'mths_since_last_delinq',
        'mths_since_last_record', 'mths_since_last_major_derog', 'hardship_flag',
        'hardship_type', 'hardship_reason', 'hardship_status', 'deferral_term',
        'hardship_amount', 'hardship_start_date', 'hardship_end_date', 'payment_plan_start_date',
        'hardship_length', 'hardship_dpd', 'hardship_loan_status',
        'orig_projected_additional_accrued_interest', 'hardship_payoff_balance_amount',
        'hardship_last_payment_amount', 'debt_settlement_flag', 'debt_settlement_flag_date',
        'settlement_status', 'settlement_date', 'settlement_amount', 'settlement_percentage',
        'settlement_term', 'funded_amnt', 'funded_amnt_inv', 'initial_list_status',
        'verification_status_joint'
    ]
    existing_cols_to_drop = [col for col in cols_to_drop if col in df_filtered.columns]
    df_cleaned = df_filtered.drop(columns=existing_cols_to_drop, errors='ignore')
    print(f"剔除贷后及无关特征后，形状为: {df_cleaned.shape}")
    good_status = ['Fully Paid']
    bad_status = ['Charged Off', 'Default']
    df_model_data = df_cleaned[df_cleaned['loan_status'].isin(good_status + bad_status)].copy()
    df_model_data['Y'] = df_model_data['loan_status'].apply(lambda x: 0 if x in good_status else 1)
    df_model_data = df_model_data.drop(columns=['loan_status'])
    print(f"定义Y并筛选后，数据集形状: {df_model_data.shape}")
    final_df = df_model_data.copy()
    print(f"使用全部 {len(final_df)} 条数据进行OOT划分。")
    print("开始进行特征工程与格式转换...")
    if 'term' in final_df.columns: final_df['term'] = final_df['term'].str.extract('(\d+)').astype(float)
    if 'int_rate' in final_df.columns: final_df['int_rate'] = final_df['int_rate'].astype(float) / 100.0
    if 'revol_util' in final_df.columns: final_df['revol_util'] = final_df['revol_util'].astype(float) / 100.0
    emp_map = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4, '5 years': 5, '6 years': 6,
               '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10}
    if 'emp_length' in final_df.columns: final_df['emp_length'] = final_df['emp_length'].map(emp_map)
    if 'earliest_cr_line' in final_df.columns: final_df['earliest_cr_line'] = pd.to_datetime(
        final_df['earliest_cr_line'], errors='coerce')
    if 'issue_d' in final_df.columns: final_df['issue_d'] = pd.to_datetime(final_df['issue_d'], errors='coerce')
    if 'earliest_cr_line' in final_df.columns and 'issue_d' in final_df.columns:
        final_df['credit_history_months'] = ((final_df['issue_d'] - final_df['earliest_cr_line']).dt.days) / 30.0
    cols_to_remove_after_processing = ['earliest_cr_line', 'zip_code', 'addr_state', 'sub_grade',
                                       'emp_title']
    final_df = final_df.drop(columns=[col for col in cols_to_remove_after_processing if col in final_df.columns])
    print("开始处理缺失值...")
    cols_to_fill_999 = ['mths_since_rcnt_il', 'mths_since_recent_bc_dlq', 'mths_since_recent_revol_delinq',
                        'mths_since_recent_inq', 'mo_sin_rcnt_rev_tl_op', 'mths_since_recent_bc', 'mo_sin_rcnt_tl']
    cols_to_fill_0 = ['il_util', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m', 'open_acc_6m', 'open_act_il',
                      'open_il_12m', 'open_il_24m', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'total_bal_il',
                      'emp_length', 'pub_rec_bankruptcies', 'collections_12_mths_ex_med', 'mo_sin_old_il_acct',
                      'num_tl_120dp_2m', 'avg_cur_bal', 'mo_sin_old_rev_tl_op', 'num_actv_rev_tl', 'num_il_tl',
                      'num_op_rev_tl', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim', 'acc_open_past_24mths',
                      'mort_acc', 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_bc_sats', 'num_bc_tl',
                      'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
                      'num_tl_op_past_12m', 'tax_liens', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit',
                      'total_il_high_credit_limit']
    cols_to_fill_median = ['dti', 'revol_util', 'credit_history_months', 'last_fico_range_high', 'last_fico_range_low',
                           'pct_tl_nvr_dlq', 'bc_open_to_buy', 'bc_util', 'percent_bc_gt_75']
    ALL_COLS_TO_KEEP = cols_to_fill_999 + cols_to_fill_0 + cols_to_fill_median
    missing_rates = final_df.isnull().sum() / len(final_df)
    high_missing_cols = missing_rates[missing_rates > 0.4].index.tolist()
    cols_to_actually_drop = [col for col in high_missing_cols if col not in ALL_COLS_TO_KEEP]
    if cols_to_actually_drop: final_df = final_df.drop(columns=cols_to_actually_drop)
    for col in cols_to_fill_999:
        if col in final_df.columns: final_df[col].fillna(999, inplace=True)
    for col in cols_to_fill_0:
        if col in final_df.columns: final_df[col].fillna(0, inplace=True)
    for col in cols_to_fill_median:
        if col in final_df.columns and final_df[col].isnull().any():
            median_val = final_df[col].median()
            final_df[col].fillna(median_val, inplace=True)
    print("缺失值填充完成。")
    print("开始处理异常值...")
    cols_for_strict_cap = ['annual_inc']
    cols_for_standard_cap = ['dti', 'revol_bal', 'tot_cur_bal', 'total_rev_hi_lim',
                             'tot_hi_cred_lim', 'total_bal_ex_mort', 'avg_cur_bal']
    for col in cols_for_strict_cap:
        if col in final_df.columns:
            upper_bound = final_df[col].quantile(0.995)
            final_df[col] = np.clip(final_df[col], a_min=None, a_max=upper_bound)
    for col in cols_for_standard_cap:
        if col in final_df.columns:
            upper_bound = final_df[col].quantile(0.99)
            final_df[col] = np.clip(final_df[col], a_min=None, a_max=upper_bound)
    all_numeric_cols = final_df.select_dtypes(include=np.number).columns.tolist()
    if 'Y' in all_numeric_cols: all_numeric_cols.remove('Y')
    if 'issue_d' in all_numeric_cols: all_numeric_cols.remove('issue_d') # 排除 'issue_d'
    processed_cols = cols_for_strict_cap + cols_for_standard_cap
    remaining_numeric_cols = [col for col in all_numeric_cols if col not in processed_cols]
    for col in remaining_numeric_cols:
        lower_bound = final_df[col].quantile(0.01)
        upper_bound = final_df[col].quantile(0.99)
        final_df[col] = np.clip(final_df[col], lower_bound, upper_bound)
    print("异常值处理完成。")
    print(f"精简前原始形状: {final_df.shape}")
    cols_to_drop_manually = [
        'collections_12_mths_ex_med', 'policy_code', 'open_acc_6m', 'open_act_il',
        'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util',
        'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi',
        'total_cu_tl', 'inq_last_12m', 'num_tl_120dpd_2m', 'num_tl_30dpd',
        'fico_range_low', 'grade', 'last_fico_range_high', 'last_fico_range_low'
    ]
    existing_cols_to_drop = [col for col in cols_to_drop_manually if col in final_df.columns]
    final_df_filtered = final_df.drop(columns=existing_cols_to_drop)
    print(f"手动移除了 {len(existing_cols_to_drop)} 个特征。")
    print(f"精简后形状 (准备OOT切分): {final_df_filtered.shape}")
    print("\n--- [阶段一-B] 执行 V4 核心特征筛选 ---")
    print("剔除所有高PSI风险的复杂征信局特征...")
    CORE_FEATURES_TO_KEEP = [
        'Y',
        'issue_d',
        'loan_amnt',             # 贷款金额
        'term',                  # 期限
        'int_rate',              # 利率 (关键风险定价)
        'installment',           # 月供
        'purpose',               # 用途
        'home_ownership',        # 住房情况
        'emp_length',            # 工作年限
        'fico_range_high',       # FICO 分
        'annual_inc',            # 年收入
        'dti',                   # DTI
        'revol_util',            # 循环额度使用率
        'verification_status',   # 验证状态
        'delinq_2yrs',           # 2年逾期
        'pub_rec_bankruptcies',  # 破产记录
        'credit_history_months'  # 信用历史
    ]
    existing_core_cols = [col for col in CORE_FEATURES_TO_KEEP if col in final_df_filtered.columns]
    print(f"原始特征数: {len(final_df_filtered.columns)}")
    final_df_filtered = final_df_filtered[existing_core_cols]
    print(f"核心模型特征数 (V4): {len(final_df_filtered.columns)}")
    print("--- [阶段一-B] V4 核心特征筛选完成 ---")
    print("--- [阶段一] 数据预处理全部完成 ---")
except FileNotFoundError:
    print("\n错误: 'accepted_2007_to_2018q4.csv' 未找到。请确保文件在脚本所在的目录中。")
print("\n--- [阶段二] 开始OOT (Out-of-Time) 切分 ---")
print("使用滚动时间窗口 (V2): 训练数据更接近测试数据，以对抗客群偏移")
train_df = final_df_filtered[
    (final_df_filtered['issue_d'] >= '2012-01-01') &
    (final_df_filtered['issue_d'] < '2013-07-01')
].copy()
val_df = final_df_filtered[
    (final_df_filtered['issue_d'] >= '2013-07-01') &
    (final_df_filtered['issue_d'] < '2014-01-01')
].copy()
test_df = final_df_filtered[
    (final_df_filtered['issue_d'] >= '2014-01-01') &
    (final_df_filtered['issue_d'] <= '2014-12-31')
].copy()
y_train = train_df['Y']
X_train = train_df.drop(columns=['Y', 'issue_d'])
y_val = val_df['Y']
X_val = val_df.drop(columns=['Y', 'issue_d'])
y_test = test_df['Y']
X_test = test_df.drop(columns=['Y', 'issue_d'])
print(f"训练集 (2012-2013H1): X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"验证集 (2013H2):     X_val:   {X_val.shape}, y_val:   {y_val.shape}")
print(f"测试集 (2014-OOT):   X_test:  {X_test.shape}, y_test:  {y_test.shape}")
if X_train.empty or X_val.empty or X_test.empty:
    print("\n*** 警告: 数据集为空！请检查 [阶段一] 的日期筛选和 [阶段二] 的切分日期是否正确！ ***")
print("\n--- [阶段三-A] 开始特征工程 (基于稳定的核心组件) ---")
def create_stable_ratio_features(df):
    df_new = df.copy()
    df_new['annual_inc_safe'] = df_new['annual_inc'].replace(0, 0.01)
    df_new.loc[:, 'loan_to_income_ratio'] = df_new['loan_amnt'] / df_new['annual_inc_safe']
    df_new.loc[:, 'installment_to_income_ratio'] = df_new['installment'] / (df_new['annual_inc_safe'] / 12)
    df_new.loc[:, 'monthly_debt'] = (df_new['dti'] * (df_new['annual_inc_safe'] / 12))
    df_new['credit_history_safe'] = df_new['credit_history_months'].replace(0, 0.01)
    df_new.loc[:, 'delinq_to_history_ratio'] = df_new['delinq_2yrs'] / df_new['credit_history_safe']
    df_new.loc[:, 'fico_x_dti'] = df_new['fico_range_high'] * df_new['dti']
    df_new = df_new.drop(columns=['annual_inc_safe', 'credit_history_safe'])
    return df_new
X_train = create_stable_ratio_features(X_train)
X_val = create_stable_ratio_features(X_val)
X_test = create_stable_ratio_features(X_test)
print(f"特征工程完成。 X_train 新形状: {X_train.shape}")
print("\n--- [阶段五] 开始模型调优与训练 ---")
print("计算样本权重 (scale_pos_weight)...")
n_good = (y_train == 0).sum()
n_bad = (y_train == 1).sum()
scale_pos_weight = n_good / n_bad
print(f"训练集中好客户数: {n_good}, 坏客户数: {n_bad}")
print(f"计算得到的 scale_pos_weight: {scale_pos_weight:.4f}")
def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'booster': 'gbtree',
        'n_jobs': -1,
        'random_state': 42,
        'scale_pos_weight': scale_pos_weight,
        'monotone_constraints': monotone_constraints_tuple,
        'enable_categorical': True,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'min_child_weight': trial.suggest_int('min_child_weight', 20, 100),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'lambda': trial.suggest_float('lambda', 0.1, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 0.1, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9)
    }
    model = xgb.XGBClassifier(
        **params,
        n_estimators=1000,
        early_stopping_rounds=50
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model.best_score
print("开始 Optuna 调优... (使用原始特征, 这可能需要几分钟)")
study = optuna.create_study(direction='maximize', study_name='xgb_Acard_Raw_OOT')
study.optimize(objective, n_trials=30, show_progress_bar=True) # (建议增加到 50-100)
print("\n调优完成！")
print(f"最佳试验 (Validation-AUC): {study.best_value:.4f}")
print("找到的最佳超参数:")
print(study.best_params)
print("\n训练最终模型 (使用最佳参数)...")
final_xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_jobs': -1,
    'random_state': 42,
    'scale_pos_weight': scale_pos_weight,
    'monotone_constraints': monotone_constraints_tuple,
    'enable_categorical': True,
    'n_estimators': 2000,
    'early_stopping_rounds': 50
}
final_xgb_params.update(study,best_params)
model_xgb_final = xgb.XGBClassifier(**final_xgb_params)
model_xgb_final.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)
print("模型训练完成。")
print(f"模型在早停机制下，最佳迭代次数为: {model_xgb_final.best_iteration}")
print("\n--- [阶段六] 模型评估 (在 2014 OOT 测试集上) ---")
y_pred_proba_oot = model_xgb_final.predict_proba(X_test)[:, 1]
auc_score_oot = roc_auc_score(y_test, y_pred_proba_oot)
print(f"OOT 测试集 ROC-AUC 评分: {auc_score_oot:.4f}")
prob_good_oot = y_pred_proba_oot[y_test == 0]
prob_bad_oot = y_pred_proba_oot[y_test == 1]
ks_stat_oot = ks_2samp(prob_good_oot, prob_bad_oot).statistic
print(f"OOT 测试集 KS 统计量: {ks_stat_oot:.4f}")
print("\n--- [阶段七] 计算 PSI (稳定性监控) ---")
y_pred_proba_train = model_xgb_final.predict_proba(X_train)[:, 1]
score_psi = calculate_psi(y_pred_proba_train, y_pred_proba_oot)
print(f"模型分 PSI (Train vs OOT): {score_psi:.4f}")
print("\n核心特征 PSI (Train vs OOT):")
feature_psi_results = {}
constrained_numeric_cols = [
    col for col in continuous_cols
    if BUSINESS_MONOTONICITY.get(col, 0) != 0 and col in X_train.columns
]
for col in constrained_numeric_cols:
    base_feature_data = X_train[col]
    comp_feature_data = X_test[col]
    f_psi = calculate_psi(base_feature_data, comp_feature_data)
    feature_psi_results[col] = f_psi
psi_series = pd.Series(feature_psi_results).sort_values(ascending=False)
print(psi_series.head(10))
print("\n--- [阶段八] 将概率转换为评分 (Scorecard) ---")
p_model = 0.5
p_real = y_train.mean()
odds_model = p_model / (1 - p_model)
odds_real = p_real / (1 - p_real)
W = odds_real / odds_model
print(f"开始概率校准：模型输出世界(P_model=50.0%) -> 真实世界(P_real={p_real:.1%})")
print(f"修正因子 W = {W:.4f}")
y_prob_uncalibrated_oot = y_pred_proba_oot
y_prob_calibrated_oot = (y_prob_uncalibrated_oot * W) / (1 - y_prob_uncalibrated_oot + y_prob_uncalibrated_oot * W)
print("OOT 测试集概率校准完成。")
def probability_to_score(prob, base_score=600, base_odds=(1/19), pdo=20):
    factor = pdo / np.log(2)
    offset = base_score - (factor * np.log(base_odds))
    prob = np.clip(prob, 1e-7, 1 - 1e-7)
    odds = prob / (1 - prob)
    score = offset - (factor * np.log(odds))
    return score.astype(int)
print(f"\n应用评分卡参数：Base Score = 650, PDO = 50")
scores_oot = probability_to_score(
    y_prob_calibrated_oot,
    base_score=650,
    base_odds=(1/19),
    pdo=50
)
df_results_oot = pd.DataFrame({
    'Actual_Y': y_test.values,
    'Prob_Model_Output': y_prob_uncalibrated_oot,
    'Prob_Calibrated': y_prob_calibrated_oot,
    'Score': scores_oot
})
print("概率转换为评分示例 (OOT测试集前10条):")
print(df_results_oot.head(10))
print("\n分数分布概览 (OOT测试集):")
print(df_results_oot['Score'].describe())
print("\n好客户 (Y=0) 的分数分布 (OOT):")
print(df_results_oot[df_results_oot['Actual_Y'] == 0]['Score'].describe())
print("\n坏客户 (Y=1) 的分数分布 (OOT):")
print(df_results_oot[df_results_oot['Actual_Y'] == 1]['Score'].describe())
print("\n(预期：好客户的平均分应远高于坏客户的平均分)")
print("\n--- XGBoost 工业级流程 (OOT + WOE + PSI) 全部完成 ---")