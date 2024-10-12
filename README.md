# gst
This project involves building a predictive model using GST (Goods and Services Tax) data to predict the target variable based on input features. The dataset provided is preprocessed to handle missing values, categorical variables, and scaling. The model used is a Random Forest Classifier.
We are tasked with creating a predictive model that accurately estimates the target variable (Y) based on given input features (X). The goal is to build, optimize, and evaluate models that generalize well on unseen test data.

The data is split into training and test sets:
Training Set (D_train): Used to train the models.
Test Set (D_test): Used to evaluate model performance on unseen data.

### Objective
To build a robust predictive model with high accuracy and performance metrics, ensuring it generalizes well on unseen data. We also aim to tune, evaluate, and combine models for optimal performance.

---

## Step-by-Step Explanation of Code and Approach

# 1. Data Preprocessing

The first step is preparing the data for machine learning models. We use pandas to load the data and drop the unnecessary ID column, which is irrelevant for modeling.
The ID column is a unique identifier that doesn't contribute to prediction, and dropping it helps to focus on the relevant features.

# 2. Preprocessing Pipeline

To prepare the features for modeling, we define a pipeline using ColumnTransformer and StandardScaler. The pipeline does the following:
Numerical columns: Missing values are imputed using the mean, and the features are scaled.
Categorical columns: Encoded using one-hot encoding to convert them into numerical format.



# 3. Hyperparameter Tuning for Individual Models*

Random Forest (Model 1)*

Random Forest is an ensemble of decision trees that improves performance by averaging multiple trees. We use GridSearchCV to optimize hyperparameters like n_estimators, max_depth, and min_samples_split.
Random Forest works well for classification tasks as it reduces overfitting and handles both continuous and categorical features effectively.

Gradient Boosting (Model 2)

Gradient Boosting is another ensemble technique that builds trees sequentially, each correcting errors made by the previous one. We also tune its hyperparameters.

Logistic Regression (Model 3)

Logistic Regression is a simple yet effective linear model for binary classification problems. We train it with default parameters.
Logistic Regression is easy to interpret and often serves as a good baseline model for classification tasks.

# 4. Combining Models Using Voting Classifier

Ensemble learning helps improve model performance by combining the strengths of multiple models. We combine different models using VotingClassifier, which allows for both hard and soft voting:

Model 4 (RF + GB): Combines random forest and is equal to the model, which is important for certain algorithms, such as neural networks.
Model 5 (RF + LR): Combines Random Forest and Logistic Regression.
Model 6 (GB + LR): Combines Gradient Boosting and Logistic Regression.
Model 7 (RF + GB + LR): Combines all three models.

# 4. Training and Validation

The model was trained on a clean, preprocessed dataset. The data pipeline helped automate preprocessing, ensuring the test data undergoes the same transformation.
No additional hyperparameter tuning was mentioned, though RandomForest  and gradient-boosting hyperparameters can be fine-tuned to improve performance.

# 5.Evaluation Metrics

- Random Forest (Model 1) achieved an accuracy of 97.65% with an AUC-ROC score of 0.9939, showing excellent performance in distinguishing between the classes.
- Gradient Boosting (Model 2) performed similarly with an accuracy of 97.62% and an AUC-ROC of 0.9938.
- Logistic Regression (Model 3), though slightly less accurate at 96.91%, provided valuable insights, especially when combined with other models.
- Ensembles (Models 4–7) improved upon individual models, with Model 4 (RF + GB) showing the highest accuracy of 97.72% and an AUC-ROC score of 0.9943.

# 6.Confusion Matrix

A confusion matrix was plotted to provide insight into the number of true/false positives and negatives.
Class 0 dominates, but the model performs well across both classes, balancing false positives and false negatives effectively.
