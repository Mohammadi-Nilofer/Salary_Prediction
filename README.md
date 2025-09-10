# Employee Salary Prediction using ANN

---
## Table of Contents

- Overview

- Features

- Tools and Technologies

- Dataset

- How to Run the Notebook

- Results and Conclusion

---

## Overview

This project focuses on building an Artificial Neural Network (ANN) model to predict employee salaries based on a synthetic dataset. The goal is to develop a robust regression model that can accurately estimate salaries by analyzing key features such as education level, years of experience, and job title.

---

## Features

- **Data Preprocessing**: Handles categorical features and scales numerical data to prepare it for model training.
- **ANN Model**: Implements a multi-layered ANN with regularization techniques like Dropout to prevent overfitting.
- **Model Training**: Utilizes callbacks such as Early Stopping to optimize the training process.
- **Performance Evaluation**: Assesses the model's accuracy using metrics like Mean Squared Error (MSE) and R-squared (R²).
- **Influential Features**: Identifies key factors that significantly impact salary predictions, including `Experience Years`, `Age`, `Education Level`, and `Department`.

---

## Tools and Technologies

This project was developed using the following tools and libraries:

* **Python**: The core programming language used for the project.
* **TensorFlow & Keras**: Used for building and training the Artificial Neural Network model.
* **Pandas**: Utilized for data loading, cleaning, and manipulation.
* **NumPy**: Used for numerical operations and array handling.
* **Scikit-learn**: Employed for data splitting, preprocessing, and model evaluation.
* **Matplotlib**: Used for data visualization, including plotting the actual vs. predicted values.
* **Jupyter Notebook / Google Colab**: The development environment for creating and running the code.

---

## Dataset

The project uses a synthetic dataset from Kaggle titled "Employer Data," which can be accessed [here](https://www.kaggle.com/datasets/gmudit/employer-data).

**Data Dictionary**:
- `Employee_ID`: Unique identifier for each employee.
- `Name`: Full name.
- `Gender`: Male or Female.
- `Age`: Age of the employee.
- `Education_Level`: Employee's education level (e.g., High School, Bachelor, Master, PhD).
- `Experience_Years`: Number of years of professional experience.
- `Department`: Business unit (e.g., HR, Engineering).
- `Job_Title`: Employee's role (e.g., Analyst, Engineer).
- `Location`: Work location (e.g., New York, Seattle).
- `Salary`: Annual salary in USD (the target variable).

---

## How to Run the Notebook

To run the notebook and reproduce the results, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Ensure you have the dataset:**
    Download `Employers_data.csv` from the Kaggle link provided above and place it in the project directory.

3.  **Install the required libraries:**
    The project uses libraries like TensorFlow, Keras, NumPy, Pandas, and Matplotlib. You can install them using pip:
    ```bash
    pip install tensorflow pandas numpy matplotlib scikit-learn
    ```

4.  **Open the notebook:**
    Use a Jupyter environment (such as Jupyter Notebook, JupyterLab, or Google Colab) to open and run `Employee_Salary_ANN.ipynb`.

---

## Results and Conclusion

The optimized ANN model successfully predicts employee salaries with high accuracy. The performance of the final model on the test dataset is as follows:

* **Mean Squared Error (MSE)**: 18.3M
* **R-squared (R²)**: ~0.991

These results indicate that the model has excellent predictive power and is a reliable tool for salary estimation. The analysis also confirmed that **Experience Years**, **Age**, **Education Level**, and **Department** are the most influential features in predicting an employee's salary.

