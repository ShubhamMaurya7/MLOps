# MLOps Unit 1 Project

## 1. Project Objective
The objective of this project is to demonstrate a standard MLOps-friendly project structure. 
The project includes dataset handling, model training, evaluation, and model saving using 
best practices followed in machine learning workflows.

## 2. Dataset Used
The dataset used in this project is a CSV file stored inside the data/ folder.
It contains input features and a target variable for training a machine learning model.

## 3. Project Structure
mlops-unit1/
│
├── data/              # Contains dataset
├── src/               # Source code for training
├── models/            # Saved trained models
├── requirements.txt   # Dependencies
└── README.md          # Project documentation

## 4. Steps to Run the Project

Step 1: Clone the repository
    git clone <repository-url>

Step 2: Navigate into the project
    cd mlops-unit1

Step 3: Install dependencies
    pip install -r requirements.txt

Step 4: Run training script
    python src/train.py

After execution, the trained model will be saved inside the models/ folder.