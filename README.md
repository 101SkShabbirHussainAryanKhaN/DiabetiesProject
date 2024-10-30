# DiabetiesProject
i am providing Readmi.md file for description so read there.
Title Page
•	Title: Linear Regression Using Gradient Descent for Diabetes Prediction
•	Author: Shabbir Hussain
•	Github Project Link : 101SkShabbirHussainAryanKhaN/DiabetiesProject: i am providing Readmi.md file for description so read there.
•	Gmail ID : 0786shabbirhussain@gamil.com
•	Date: 30 / October / 2024
•	Institution: University of Baltistan, Skardu.
________________________________________
Abstract
This report presents the implementation of a custom Linear Regression model using Gradient Descent (GD) to predict diabetes progression based on 10 medical factors. By comparing our custom GD-based linear regression to a standard linear regression model from scikit-learn, we evaluate model performance in predicting diabetes progression. The dataset used is the well-known Diabetes Dataset from the scikit-learn library.
________________________________________
1. Introduction
Diabetes is a chronic disease that impacts millions worldwide. Predicting its progression using machine learning models offers a promising approach to early intervention and improved patient outcomes. The aim of this project is to build a custom linear regression model optimized with gradient descent and to analyze the model's performance against a standard linear regression approach.
Objectives
1.	Implement and train a custom linear regression model using gradient descent.
2.	Compare the custom model's performance with the scikit-learn's Linear Regression.
3.	Assess the interpretability of model parameters and analyze the feature coefficients.
Tools and Libraries
•	Python: Core language for implementation.
•	Numpy: Efficient numerical computations.
•	Scikit-learn: Standard machine learning library used for dataset and evaluation.
________________________________________
2. Methodology
2.1 Dataset
The Diabetes Dataset used here is from scikit-learn, consisting of 442 samples with 10 features. The target variable represents disease progression, a continuous numerical variable.
•	Features (X): 10 real-valued features representing medical information.
•	Target (y): A single continuous variable indicating the diabetes progression measure.
2.2 Data Splitting
Data is split into training (80%) and testing (20%) sets. A random seed (random_state=50) is used to ensure reproducibility.
2.3 Models
1.	Standard Linear Regression: scikit-learn’s LinearRegression class is used as a baseline model.
2.	Custom Linear Regression with Gradient Descent: A custom gradient descent-based linear regression model, GDRegressor, is implemented.
________________________________________
3. Model Implementation
3.1 Standard Linear Regression
The LinearRegression model from scikit-learn was used to fit the data, calculate the coefficients, and evaluate the model on the test set using the R² score.
3.2 Gradient Descent Regressor (GDRegressor)
Design of GDRegressor
The GDRegressor class defines the following parameters and methods:
•	Attributes:
o	coef_: Model coefficients.
o	intercept_: Model intercept.
o	lr: Learning rate, default set to 0.01.
o	epochs: Number of training iterations, default set to 100.
•	Methods:
o	fit(X_train, y_train): Implements gradient descent to optimize coefficients and intercept iteratively.
o	predict(X_test): Uses the trained model to predict on the test set.
Training Process
In each epoch, predictions (y_hat) are generated, and partial derivatives of the intercept and coefficients are calculated. The coefficients and intercept are updated iteratively using the learning rate, refining the model’s predictive accuracy. For this project, 100 epochs were set to balance convergence and efficiency.
________________________________________
4. Results
4.1 Model Coefficients and Intercept
Upon fitting the model, the coefficients and intercept values for both models are as follows:
scikit-learn Linear Regression
•	Coefficients: [41.48, -256.08, 498.42, …] (10 coefficients for each feature)
•	Intercept: 151.78
GDRegressor Linear Regression
•	Coefficients: [1.017, 1.001, 1.043, …] (Initial values show convergence towards optimal values)
•	Intercept: Constantly updated in each epoch to minimize error.
4.2 Model Performance
Using the R² Score on the test set as a performance measure:
•	Standard Linear Regression: R² = 0.5234
•	GDRegressor: The performance converges gradually as epochs increase, with the R² score improving with more epochs.
4.3 Observations
•	The learning rate (0.01) and epochs (100) impacted the speed of convergence.
•	Higher epochs allowed GDRegressor to converge closer to the standard linear regression model's R² score, indicating effective parameter optimization.
________________________________________
5. Discussion
Advantages of Gradient Descent
1.	Flexibility: GD-based models allow fine-tuning learning rates and epochs, making them suitable for various datasets.
2.	Interpretability: The iterative parameter updates provide insight into model learning.
Limitations
•	Computation Time: Compared to scikit-learn's direct linear regression fitting, gradient descent requires more computation.
•	Convergence: Requires careful selection of hyperparameters to ensure optimal convergence without overshooting.
Comparison with scikit-learn Linear Regression
While scikit-learn's linear regression is faster, the GDRegressor's flexibility makes it preferable when customizing learning processes is necessary.
________________________________________
6. Conclusion
This project implemented a custom linear regression model optimized via gradient descent, achieving comparable performance to scikit-learn's linear regression model. The custom model's iterative nature offered valuable insights into how coefficients and intercepts are optimized over time. Future enhancements could include testing more advanced optimization techniques, such as stochastic gradient descent, to further improve computational efficiency.
________________________________________
7. Future Work
1.	Explore other optimization methods (e.g., stochastic gradient descent or mini-batch gradient descent) for faster convergence.
2.	Hyperparameter Tuning: Experiment with learning rates and epochs to optimize the model further.
3.	Model Comparison: Integrate more complex regression models, such as ridge or lasso regression, to control overfitting.
________________________________________
8. References
1.	Scikit-learn documentation: https://scikit-learn.org/stable/
2.	Articles on Gradient Descent optimization in machine learning


