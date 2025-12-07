# 🚀 Spotify Data Mission

A gamified Data Science workshop that guides students through the Hierarchy of Needs in Data Science: Collection, Preparation, Mining, and Analytics.

This application uses **Flask** and **Plotly** to create an interactive Mission Control where users train an AI model to predict Spotify hits.

<img width="990" height="641" alt="dashboard" src="https://github.com/user-attachments/assets/69053cfa-323e-486c-a7d8-579bd2af3575" />

## 📊 Dataset Acknowledgement

The analysis in this project is based on the **Spotify Global Music Dataset (2009–2025)**.  
**Source:** [Kaggle - Spotify Global Music Dataset](https://www.kaggle.com/datasets/wardabilal/spotify-global-music-dataset-20092025/data)  
**Author:** Bilal Warda  
**License:** See the Kaggle page for license details.

*Note: The file `spotify_data clean.csv` in this repository is a processed version of the original dataset prepared for educational use.*

## 🛠️ Installation

### 1. Clone the Repository
Open your terminal and run:
```bash
git clone https://github.com/brunobastosrodrigues/spotify-data-mission.git
cd spotify-data-mission
```

### 2. Install Dependencies
Make sure you have Python 3 installed. Then run:
```bash
pip install -r requirements.txt
```

## 🚀 How to Run

Start the server:
```bash
python3 app.py
```

Open the application in your browser:
```
http://localhost:5000
```
If running on a virtual machine, replace `localhost` with your VM’s IP address.

## 🗺️ Mission Workflow

📥 **Collection**: Load the raw CSV data into memory.  
🧹 **Preparation**: Apply imputation to fix missing values.  
⛏️ **Mining**: Train a Logistic Regression model to predict popularity.  
📊 **Insights**: Evaluate the model using precision, recall, and F1-score visualizations.



