# 🛒 Instacart Demand Sensing & Customer Segmentation

**End-to-end machine learning pipeline** to predict product reorders and segment customers using behavioral insights from the [Instacart Online Grocery Basket Dataset](https://www.instacart.com/datasets/grocery-shopping-2017). This project combines predictive modeling, and customer segmentation to support targeted marketing and operational optimization.

---

## 📌 Problem Statement

**Goal 1: Demand Sensing**
> Predict whether a user will reorder a given product in their next purchase.

**Goal 2: Customer Segmentation**
> Cluster customers based on purchasing behavior to support personalization strategies.

---
## 📚 Dataset

### 🔗 [Instacart Market Basket Analysis Dataset](https://www.instacart.com/datasets/grocery-shopping-2017)

> This dataset, originally released for a Kaggle competition, contains real-world grocery order data from Instacart, a popular online grocery delivery service. It includes over **3 million orders** from **200,000+ customers**, covering **50,000+ products**, grouped by **21 departments** and **134 aisles**.

### 📦 Key Components:
- `orders.csv`: Metadata about each order (order number, user, eval set, etc.)
- `order_products__prior.csv`: Historical user-product purchase behavior
- `order_products__train.csv`: Training labels indicating reorders
- `products.csv`: Product names and metadata
- `aisles.csv` & `departments.csv`: Category groupings for products

### 🧠 Why This Dataset?
- Represents large-scale user behavior in e-commerce
- Enables analysis of:
  - Product reorder prediction (demand sensing)
  - User segmentation and retention
  - Purchase patterns over time
- Suitable for modeling **customer intelligence**, **recommendation systems**, and **retail analytics**

## 🧠 Project Workflow

### 1. 📦 Import Packages and Datasets
- All original Instacart CSVs loaded and merged (`orders.csv`, `products.csv`, `order_products__prior.csv`, etc.)

### 2. 📊 Initial EDA
- Distribution plots for orders, reorders, user activity, product popularity
- Checked for missing and duplicate data

### 3. 🧹 Data Cleaning
- Removed unused or empty columns
- Filtered null values, cleaned data types

### 4. 📈 Visual EDA
- Seaborn and Matplotlib plots for:
  - Top reordered products
  - Reorder ratios over time
  - Basket size and order frequency trends

### 5. 🧪 Feature Engineering
- Engineered over 15+ features including:
  - `user_reorder_ratio`, `total_orders`, `product_reorder_rate`
  - Time-based and basket-level insights
- Merged multiple files to construct the final training set

### 6. 🤖 Modeling
- **Random Forest**
- **XGBoost**
- **Logistic Regression**
- Dealt with class imbalance using **RandomUnderSampler**
- Evaluated with Accuracy, ROC-AUC, Confusion Matrix

### 7. 🏆 Model Comparison
- Compared all models with and without tuning
- **XGBoost** delivered best performance (~89% ROC-AUC)

### 8. 👥 Customer Segmentation
- Aggregated user-level behavior:
  - `reorder_ratio`, `avg_days_between_orders`, `unique_products_ordered`
- Standardized features and applied **KMeans clustering**
- Visualized segments using PCA

---

## 📌 Key Results

- **XGBoost ROC-AUC**: 0.89  
- **Top Reorder Drivers**:
  - User reorder ratio
  - Product reorder probability
  - Days since prior order
- **Customer Segments Identified**:
  - Loyal frequent buyers
  - Bulk occasional buyers
  - Infrequent one-time shoppers

---

## 💼 Business Insights

| Segment | Behavior | Suggested Strategy |
|---------|----------|--------------------|
| High reorder + frequent | Loyal customers | Offer loyalty rewards |
| Low reorder + low frequency | At-risk users | Retargeting via email or discounts |
| High product variety | Exploratory users | Personalized recommendations |

---

## 🛠 Tech Stack

- **Languages**: Python 3.10
- **Libraries**: pandas, numpy, scikit-learn, xgboost, shap, matplotlib, seaborn
- **Modeling**: Random Forest, XGBoost, Logistic Regression
- **Clustering**: KMeans, PCA
- **Platform**: Jupyter Notebook (local + Colab-compatible)

---

## 📁 File Structure

```bash
.
├── Demand_sensing.ipynb       # Main notebook
├── README.md                  # Project overview
├── data/                      # Raw CSVs from Instacart
└── images/                    # Visuals (optional)
