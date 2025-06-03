# AI Fraud Detection System

This project implements a machine learning system to detect fraudulent credit card transactions using the "Credit Card Fraud Detection" dataset from Kaggle. It addresses the common challenge of highly imbalanced data in fraud detection scenarios.

![Project Flow Diagram](path/to/your/flow_diagram.jpg)
*(Replace `path/to/your/flow_diagram.jpg` with the actual path or URL to the flow diagram image in your repository if you've uploaded it.)*

## Table of Contents
1.  [Introduction](#introduction)
2.  [Dataset](#dataset)
3.  [Project Workflow](#project-workflow)
4.  [Technologies Used](#technologies-used)
5.  [Setup and Installation](#setup-and-installation)
6.  [Usage](#usage)
7.  [Model Evaluation Metrics](#model-evaluation-metrics)
8.  [Results Summary](#results-summary)
9.  [Future Work](#future-work)
10. [License](#license)
11. [Acknowledgements](#acknowledgements)

## Introduction
Credit card fraud is a significant concern for financial institutions and consumers. This project aims to build an effective AI-powered system to identify potentially fraudulent transactions. The primary challenge is the highly imbalanced nature of the dataset, where fraudulent transactions are a very small minority. Techniques like SMOTE (Synthetic Minority Over-sampling Technique) are employed to address this.

## Dataset
The dataset used is `creditcard.csv`, which contains transactions made by European cardholders in September 2013 over two days.

*   **Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (or mention if sourced elsewhere)
*   **Features:**
    *   `Time`: Seconds elapsed between this transaction and the first transaction in the dataset.
    *   `V1`-`V28`: Anonymized features, likely results of PCA transformation.
    *   `Amount`: Transaction amount.
    *   `Class`: Target variable (1 for fraud, 0 for non-fraud).
*   **Characteristics:** Highly imbalanced, with frauds accounting for approximately 0.172% of all transactions.

## Project Workflow
The project follows these main steps:
1.  **Data Loading & Inspection:** Load the dataset and perform initial exploratory data analysis (EDA) to understand its structure, missing values, and basic statistics.
2.  **Data Preprocessing:**
    *   Scale numerical features (`Time`, `Amount`) using `StandardScaler`.
3.  **Exploratory Data Analysis (Continued):**
    *   Visualize the class distribution to highlight the imbalance.
4.  **Data Splitting:** Split the data into training and testing sets, using stratification to maintain class proportions.
5.  **Handling Class Imbalance:** Apply SMOTE (Synthetic Minority Over-sampling Technique) to the *training data only* to create a balanced dataset for model training.
6.  **Model Training:**
    *   Train Logistic Regression on the SMOTE-balanced data.
    *   Train Random Forest Classifier on the SMOTE-balanced data.
    *   Train Logistic Regression with `class_weight='balanced'` on the original imbalanced training data as an alternative.
7.  **Model Evaluation:** Evaluate the trained models on the unseen test set using appropriate metrics for imbalanced classification.
8.  **Interpretation & Conclusion:** Analyze results, compare model performances, and draw conclusions.

## Technologies Used
*   **Python 3.x**
*   **Libraries:**
    *   `pandas`: For data manipulation and CSV file I/O.
    *   `numpy`: For numerical operations.
    *   `scikit-learn`: For machine learning tasks (model building, preprocessing, metrics).
    *   `imbalanced-learn`: For SMOTE and other imbalance handling techniques.
    *   `matplotlib`: For basic plotting.
    *   `seaborn`: For enhanced statistical visualizations.
    *   `Jupyter Notebook` (Optional, for experimentation and running the script interactively)

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    A `requirements.txt` file should be included.
    ```bash
    pip install -r requirements.txt
    ```
    If `requirements.txt` is not provided, you can install them manually:
    ```bash
    pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn jupyterlab
    ```

4.  **Dataset:**
    Ensure the `creditcard.csv` file is present in the root directory of the project or update the path in the script accordingly.

## Usage
To run the fraud detection pipeline, execute the main Python script (e.g., `fraud_detection_script.py` or the name of your main Jupyter Notebook):

```bash
python your_main_script_name.py
