# 🧭 員工離職分析與預測（Employee Attrition Analysis & Prediction）

## 🇨🇳 中文版說明

### 📖 專案簡介
本專案透過資料分析與機器學習模型，探討員工離職（Attrition）的關鍵因素。  
利用 **統計檢定（ANOVA、卡方）**、**相關係數熱圖（Correlation Heatmap）** 及 **決策樹（Decision Tree）** 等方法，  
從不同角度理解哪些特徵最能影響員工離職的可能性。

---

### 🔍 分析步驟
1. **資料清理與前處理**
   - 缺失值處理、編碼（如 `pd.get_dummies()`）
   - 標準化數值型變數  

2. **探索性資料分析（EDA）**
   - 以 **相關係數熱圖 (heatmap)** 檢視特徵與 Attrition 的關聯性  
   - 使用 **ANOVA / 卡方檢定 (Chi-square)** 驗證變數與離職之間的顯著性  

3. **模型建構與視覺化**
   - 建立 **決策樹 (Decision Tree)** 模型  
   - 使用 **混淆矩陣 (Confusion Matrix)** 評估模型準確度  
   - 視覺化決策邏輯（離職關鍵特徵居中顯示）

---

### 📊 主要發現
| 變數名稱 | 結果 | 顯著性 | 影響力 | 說明 |
|-----------|--------|------------|------------|--------|
| MonthlyIncome | 負相關 | ✅ p<0.05 | ★★★☆☆ | 薪資越低越容易離職 |
| JobSatisfaction | 負相關 | ✅ p<0.05 | ★★★★☆ | 滿意度越低越容易離職 |
| DistanceFromHome | 正相關 | ✅ p<0.05 | ★★★☆☆ | 距離越遠越容易離職 |
| JobLevel | 負相關 | ✅ p<0.05 | ★★★☆☆ | 職等越高越穩定 |
| WorkLifeBalance | 負相關 | ✅ p<0.05 | ★★☆☆☆ | 工作與生活平衡差 → 離職率高 |

---

### 🧠 結論
綜合 ANOVA、Heatmap、Decision Tree 結果：
- **薪資、工作滿意度、工作年資** 為影響離職的主要因素  
- **工作地點距離與工作壓力** 也是潛在風險  
- 建議企業優化薪資結構與員工發展機會，以降低流動率  

---

### ⚙️ 環境需求
```bash
Python >= 3.8
pandas, numpy, seaborn, matplotlib, scikit-learn
```

---

### 💻 執行方式
```bash
# 安裝套件
pip install -r requirements.txt

# 執行分析
python analysis.py
```

---

### 🧑‍💻 作者
**Po-Kuang (Clément) Chen**  
- 🎓 Master in Big Data & AI, France  
- 💼 專長：Time Series Forecasting / Data Analysis / Machine Learning  
- 🔗 [LinkedIn](https://www.linkedin.com/in/po-kuang-chen-23625821a/) | [Website](https://poseuropelife.com/aboutme/)

---

## 🇺🇸 English Version

### 📖 Overview
This project explores the key factors influencing **employee attrition** using statistical and machine learning approaches.  
By combining **ANOVA**, **Chi-square tests**, **Correlation Heatmaps**, and **Decision Tree models**,  
we aim to understand which variables most strongly impact employees’ decisions to leave.

---

### 🔍 Workflow
1. **Data Cleaning & Preprocessing**
   - Handle missing values and categorical encoding (`pd.get_dummies()`)
   - Normalize numerical features

2. **Exploratory Data Analysis (EDA)**
   - Visualize correlations between features and attrition via heatmap  
   - Conduct **ANOVA / Chi-square tests** to check statistical significance  

3. **Modeling & Visualization**
   - Build a **Decision Tree** classifier  
   - Evaluate performance using **Confusion Matrix**  
   - Visualize key features in the decision process  

---

### 📊 Key Insights
| Feature | Correlation | Significance | Impact | Interpretation |
|----------|--------------|---------------|----------|----------------|
| MonthlyIncome | Negative | ✅ p<0.05 | ★★★☆☆ | Lower income → higher attrition |
| JobSatisfaction | Negative | ✅ p<0.05 | ★★★★☆ | Lower satisfaction → higher attrition |
| DistanceFromHome | Positive | ✅ p<0.05 | ★★★☆☆ | Longer commute → higher attrition |
| JobLevel | Negative | ✅ p<0.05 | ★★★☆☆ | Higher job level → more stable |
| WorkLifeBalance | Negative | ✅ p<0.05 | ★★☆☆☆ | Poor balance → higher attrition |

---

### 🧠 Conclusion
Based on ANOVA, correlation, and decision tree analysis:
- **Income, satisfaction, and experience** are the strongest predictors of attrition.  
- **Commute distance and stress levels** also contribute to turnover risk.  
- Companies should enhance compensation fairness and employee development to reduce attrition.

---

### ⚙️ Environment
```bash
Python >= 3.8
pandas, numpy, seaborn, matplotlib, scikit-learn
```

---

### 💻 How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis
python analysis.py
```

---

### 👤 Author
**Po-Kuang (Clément) Chen**  
- 🎓 Master’s in Big Data & AI (France)  
- 💼 Expertise: Time Series Forecasting, Machine Learning, Data Analysis  
- 🔗 [LinkedIn](https://www.linkedin.com/in/po-kuang-chen-23625821a/) | [Website](https://poseuropelife.com/aboutme/)
