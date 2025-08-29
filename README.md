# Diabetes-Classification-with-Machine-Learning
This project applies machine learning techniques to classify whether a patient has diabetes based on medical attributes. It uses feature engineering, data preprocessing, and a Random Forest classifier pipeline to deliver insights and predictions.
⚙️ Requirements

Install dependencies using:

pip install -r requirements.txt


Contents of requirements.txt:

numpy

pandas

scikit-learn

seaborn

matplotlib

🚀 How to Run

Clone the repository:

git clone https://github.com/omarfoud/Diabetes-Classification-with-Machine-Learning.git
cd Diabetes-Classification-with-Machine-Learning


Run the script:

python diabetes_classification.py


Outputs:

Console prints:

Class distribution

Accuracy score

Classification report

Cross-validation accuracy

A feature importance plot saved as:

feature_importance.png

🔬 Approach

Data Cleaning: Replaces zeros in key medical features with NaN and imputes missing values.

Feature Engineering: Creates BMI categories and glucose level bins.

Preprocessing: Uses ColumnTransformer with scaling and imputation.

Model: Random Forest with tuned hyperparameters (n_estimators=200, max_depth=10, etc.).

Feature Selection: Selects top 10 features using ANOVA F-test (SelectKBest).

Evaluation: Accuracy, classification report, feature importance visualization, cross-validation.

📊 Example Output

Accuracy: ~0.80 (varies by split)

Feature Importance Plot:


📌 Dataset

The dataset used is the Pima Indians Diabetes Dataset, which contains diagnostic measurements for women aged 21+ of Pima Indian heritage.

Features: Glucose, BMI, Blood Pressure, Insulin, Age, etc.

Target: Outcome (1 = diabetes, 0 = no diabetes).

🤝 Contribution

Pull requests and improvements are welcome!
