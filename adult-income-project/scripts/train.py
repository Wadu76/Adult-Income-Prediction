import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb

#1. 数据加载与基础清洗
def load_data(train_path, test_path):
    """加载训练集和测试集，并处理基础格式问题"""
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
    ]
    
    #加载训练集
    train = pd.read_csv(train_path, names=columns, na_values='?', skipinitialspace=True)
    
    #加载测试集（跳过首行，并处理income列的格式）
    test = pd.read_csv(test_path, names=columns, na_values='?', skipinitialspace=True, skiprows=1)
    test['income'] = test['income'].str.replace('.', '')  #移除测试集income中的 '.' 

    #删除冗余特征
    for df in [train, test]:
        df.drop('fnlwgt', axis=1, inplace=True)
    
    return train, test


#2. 数据预处理
def preprocess_data(train, test):
    """处理缺失值、编码分类变量、标准化数值特征"""
    #定义分类和数值特征列
    categorical_features = ['workclass', 'education', 'marital_status', 'occupation',
                            'relationship', 'race', 'sex', 'native_country']
    numerical_features = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    
    #使用训练集的众数填充缺失值（避免链式赋值） 被panda组件提醒过，无伤大雅不过还是改了一下
    for col in categorical_features:
        mode_val = train[col].mode()[0]
        #直接对原始DataFrame的列赋值
        train[col] = train[col].fillna(mode_val)
        test[col] = test[col].fillna(mode_val)
    
    #构建预处理Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    #在训练集上拟合预处理器
    preprocessor.fit(train)
    
    #转换训练集和测试集
    X_train = preprocessor.transform(train)
    X_test = preprocessor.transform(test)
    
    #提取目标变量
    y_train = train['income'].map({'<=50K': 0, '>50K': 1})
    y_test = test['income'].map({'<=50K': 0, '>50K': 1})
    
    return X_train, X_test, y_train, y_test, categorical_features, numerical_features

#3.报告部分，计算
def chinese_classification_report(y_true, y_pred):
    #计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    #计算各项指标
    tn, fp, fn, tp = cm.ravel()
    
    #准确率 = (TP + TN) / (TP + TN + FP + FN)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    #精确率 = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    #召回率 = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    #F1分数 = 2 * (精确率 * 召回率) / (精确率 + 召回率)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    #负类精确率和召回率
    neg_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    neg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    neg_f1 = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall) if (neg_precision + neg_recall) > 0 else 0
    
    #构建报告
    report = f"""
分类报告:
---------------------------------------
              精确率    召回率    F1分数
---------------------------------------
收入<=50K     {neg_precision:.4f}    {neg_recall:.4f}    {neg_f1:.4f}
收入>50K      {precision:.4f}    {recall:.4f}    {f1:.4f}
---------------------------------------
准确率: {accuracy:.4f}
混淆矩阵:
[[{tn} {fp}]
 [{fn} {tp}]]
"""
    return report


#4. 模型训练与纵向评估
def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        '逻辑回归': LogisticRegression(max_iter=1000, class_weight='balanced'),
        '决策树': DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42),
        '随机森林': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        print(f"正在训练 {name} 模型...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            'classification_report': chinese_classification_report(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba),
            'model': model  #返回训练好的模型
        }
    
    return results


#5. xgboooooost特征重要性分析
def analyze_feature_importance(train_data, categorical_features, numerical_features):
    """分析特征重要性"""
    #准备数据
    X = train_data.drop('income', axis=1)
    y = train_data['income'].map({'<=50K': 0, '>50K': 1})
    
    #使用随机森林分析特征重要性
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    #对数值特征进行标准化
    X_num = X[numerical_features].copy()
    for col in numerical_features:
        X_num[col] = (X_num[col] - X_num[col].mean()) / X_num[col].std()
    
    #对分类特征进行独热编码
    X_cat = pd.get_dummies(X[categorical_features], drop_first=False)
    
    #合并特征
    X_processed = pd.concat([X_num, X_cat], axis=1)
    
    #训练模型
    rf.fit(X_processed, y)
    
    #获取特征重要性
    importances = rf.feature_importances_
    
    #创建特征重要性DataFrame
    feature_importance = pd.DataFrame({
        'feature': X_processed.columns,
        'importance': importances
    })
    
    #按重要性排序
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    return feature_importance.head(10)  #返回前10个重要特征


#主程序
if __name__ == "__main__":
    #加载数据（确保是绝对路径）
    train_data, test_data = load_data('C:/Users/Wadu76/Desktop/adult-income-project/data/adult.data',
     'C:/Users/Wadu76/Desktop/adult-income-project/data/adult.test')
    
    print("数据加载完成。")
    print(f"训练集大小: {train_data.shape}")
    print(f"测试集大小: {test_data.shape}")
    
    #预处理
    X_train, X_test, y_train, y_test, categorical_features, numerical_features = preprocess_data(train_data, test_data)
    print("数据预处理完成。")
    
    #训练与评估
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    #打印结果
    print("\n模型评估结果：")
    for model_name, metrics in results.items():
        print(f"\n=== {model_name} ===")
        print(metrics['classification_report'])
        print(f"AUC曲线下面积: {metrics['auc']:.4f}")
    
    #分析特征重要性
    try:
        print("\n特征重要性分析：")
        feature_importance = analyze_feature_importance(train_data, categorical_features, numerical_features)
        
        print("\n前10个重要特征:")
        for i, (feature, importance) in enumerate(zip(feature_importance['feature'], feature_importance['importance'])):
            print(f"{i+1}. {feature}: {importance:.4f}")
        
        #直接使用XGBoost模型的特征重要性
        print("\nXGBoost模型特征重要性：")
        
        #获取XGBoost模型
        xgb_model = results['XGBoost']['model']
        
        #准备数据（与上面相同的方式）
        X_num = train_data[numerical_features].copy()
        for col in numerical_features:
            X_num[col] = (X_num[col] - X_num[col].mean()) / X_num[col].std()
        
        X_cat = pd.get_dummies(train_data[categorical_features], drop_first=False)
        X_processed = pd.concat([X_num, X_cat], axis=1)
        
        #训练新的XGBoost模型（因为之前的模型使用的是转换后的特征）
        xgb_model_new = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        xgb_model_new.fit(X_processed, train_data['income'].map({'<=50K': 0, '>50K': 1}))
        
        #获取特征重要性
        xgb_importances = xgb_model_new.feature_importances_
        
        #创建特征重要性DataFrame
        xgb_feature_importance = pd.DataFrame({
            'feature': X_processed.columns,
            'importance': xgb_importances
        })
        
        #按重要性排序
        xgb_feature_importance = xgb_feature_importance.sort_values('importance', ascending=False)
        
        print("\n前10个重要特征（基于XGBoost）:")
        for i, (feature, importance) in enumerate(zip(xgb_feature_importance['feature'].head(10), 
                                                     xgb_feature_importance['importance'].head(10))):
            print(f"{i+1}. {feature}: {importance:.4f}")
            
    except Exception as e:
        print(f"特征重要性分析失败: {e}")
        
    print("\n模型训练与评估完成！")
