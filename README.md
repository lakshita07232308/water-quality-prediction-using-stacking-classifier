# Water Quality Prediction using an Ensemble Machine Learning Model

This project presents a robust machine learning framework for predicting water potability. It addresses the limitations of individual classifier models by implementing an ensemble-based **Stacking Classifier** that integrates Naive Bayes, Decision Tree, and MLP models. To enhance trust and transparency, the model's predictions are explained using **SHAP (SHapley Additive exPlanations)**.

## Problem Statement

Traditional methods for water quality assessment are often slow and expensive. While machine learning offers a faster alternative, single-model approaches have significant drawbacks:
* **Naive Bayes** assumes that features (like pH and turbidity) are independent, which is often not true for chemical parameters in water.
* **Decision Trees** are prone to overfitting, making them less reliable on new, unseen data.
* **Multilayer Perceptrons (MLPs)** can be difficult to optimize and often act as "black boxes," making their predictions hard to interpret.

This project aims to overcome these challenges by creating a more accurate, robust, and interpretable prediction model.

## Proposed Solution

To address the gaps identified, our solution is a multi-layered framework:

1.  **Ensemble Learning:** We use a **Stacking Classifier** that combines the strengths of three diverse base models (Naive Bayes, Decision Tree, MLP) and uses a Logistic Regression meta-classifier to make the final prediction. This approach mitigates the individual weaknesses of each model, leading to higher accuracy.
2.  **Model Interpretability:** We apply **SHAP** to our final model. This allows us to quantify the impact of each water quality parameter (e.g., pH, hardness, turbidity) on the final prediction, turning our "black box" into a transparent and trustworthy tool.
3.  **Robust Preprocessing:** A standardized data preprocessing pipeline is used, which includes **median imputation** for handling missing values and **Min-Max scaling** to normalize the data.

## Key Features

* **High Accuracy:** Achieves superior performance by leveraging an ensemble of machine learning models.
* **Transparent Predictions:** Uses SHAP to provide clear, actionable insights into which factors are most critical for determining water quality.
* **Robust Data Handling:** Implements a consistent preprocessing pipeline to ensure data integrity and model reliability.
* **Deployable Model:** The final trained model is serialized into a `.pkl` file, making it ready for integration into real-time monitoring systems.

## Technology Stack

* **Language:** Python
* **Libraries:**
    * Pandas: For data manipulation and preprocessing.
    * Scikit-learn: For building and evaluating machine learning models (Naive Bayes, Decision Tree, MLP, Stacking Classifier).
    * SHAP: For model explainability and feature importance analysis.
    * Joblib: For serializing the final model.

## Dataset

The model was trained on the **Water Quality Dataset**, which contains various physicochemical parameters to determine potability. Key features include:
* pH
* Hardness
* Solids (Total Dissolved Solids)
* Chloramines
* Sulfate
* Conductivity
* Organic Carbon
* Trihalomethanes
* Turbidity

## Results

The Stacking Classifier demonstrated superior performance compared to the standalone models across all key metrics.

| Model           | Accuracy | Precision | Recall | F1-Score |
| :-------------- | :------: | :-------: | :----: | :------: |
| Naive-Bayes     |  61.4%   |   0.63    |  0.89  |   0.74   |
| Decision Tree   |  63.4%   |   0.65    |  0.87  |   0.74   |
| MLP             |  65.1%   |   0.67    |  0.85  |   0.75   |
| **Stacking Model** | **67.22%** |  **0.70** | **0.91** |  **0.77** |

The SHAP analysis identified **pH, turbidity, and conductivity** as the most influential factors in predicting water potability.

## How to Run the Project

To get this project running on your local machine, follow these steps.

### Prerequisites
* Python 3.8+
* pip (Python package installer)

### Installation & Execution
1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/water-quality-prediction.git](https://github.com/your-username/water-quality-prediction.git)
    cd water-quality-prediction
    ```
2.  **Install the required dependencies:**
    ```sh
    pip install pandas scikit-learn shap joblib
    ```
3.  **Run the main script:**
    * Place your water quality dataset (e.g., `water_quality.csv`) in the project directory.
    * Execute the Python script to preprocess the data, train the models, and save the final `.pkl` file.
    ```sh
    python train_model.py
    ```

---
*This project was submitted as part of the BITE410L - Machine Learning course at Vellore Institute of Technology.*
