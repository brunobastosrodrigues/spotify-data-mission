from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.utils
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.secret_key = 'datascience_is_cool'

# Global storage for the loaded dataset
DATASET = None

@app.route('/')
def index():
    return render_template('index.html')

# --- STEP 1: COLLECTION ---
@app.route('/step1', methods=['GET', 'POST'])
def step1_collection():
    global DATASET
    if request.method == 'POST':
        try:
            DATASET = pd.read_csv('spotify_data clean.csv')
            flash(f"Success! {len(DATASET)} songs collected from the CSV file.", "success")
            return redirect(url_for('step2_preparation'))
        except FileNotFoundError:
            flash("Error: File not found. Make sure 'spotify_data clean.csv' is in the folder.", "danger")
    
    return render_template('step1_collection.html')

# --- STEP 2: PREPARATION ---
@app.route('/step2', methods=['GET', 'POST'])
def step2_preparation():
    global DATASET
    if DATASET is None: return redirect(url_for('step1_collection'))
    
    # Analytics: Count missing values before fixing
    missing_before = DATASET.isnull().sum().sum()
    
    if request.method == 'POST':
        try:
            # Smart Repair: Only fix columns if they exist
            target_cols = ["artist_name", "artist_genres"]
            existing_cols = [col for col in target_cols if col in DATASET.columns]
            
            for col in existing_cols:
                # Mode Imputation (Slide 36)
                if not DATASET[col].mode().empty:
                    DATASET[col] = DATASET[col].fillna(DATASET[col].mode()[0])
                else:
                    DATASET[col] = DATASET[col].fillna("Unknown")
            
            flash("Data scrubbed! Missing values imputed using the Mode.", "success")
            return redirect(url_for('step3_mining'))
            
        except Exception as e:
            flash(f"Error during repair: {str(e)}", "danger")
            return redirect(url_for('step2_preparation'))

    return render_template('step2_preparation.html', missing_count=missing_before, 
                           preview=DATASET.head(5).to_html(classes='table table-sm table-hover', index=False))

# --- STEP 3: MINING (Configuration) ---
@app.route('/step3', methods=['GET', 'POST'])
def step3_mining():
    if DATASET is None: return redirect(url_for('step1_collection'))

    # Analytics: Only show numeric columns as potential features
    numeric_cols = DATASET.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if request.method == 'POST':
        session['target'] = request.form.get('target', 'track_popularity')
        session['features'] = request.form.getlist('features')
        session['threshold'] = float(request.form.get('threshold', 75))
        return redirect(url_for('step4_results'))

    return render_template('step3_mining.html', columns=numeric_cols)

# --- STEP 4: ANALYTICS & VISUALIZATION ---
@app.route('/step4')
def step4_results():
    if DATASET is None: return redirect(url_for('step1_collection'))
    
    # Retrieve user choices
    target_col = session.get('target', 'track_popularity')
    feature_cols = session.get('features', ['track_duration_min', 'artist_popularity'])
    threshold = session.get('threshold', 75)
    
    # 1. Prepare Data
    df_clean = DATASET.copy()
    
    # Target Engineering: Create Binary Target (Hit vs Miss)
    binary_target = "is_hit"
    df_clean[binary_target] = (df_clean[target_col] > threshold).astype(int)
    
    # Drop rows with missing values in selected features
    df_clean = df_clean.dropna(subset=feature_cols + [binary_target])
    
    X = df_clean[feature_cols]
    y = df_clean[binary_target]
    
    # Analytics Safety Check: Do we have both Hits and Misses?
    if y.nunique() < 2:
        flash(f"Analysis failed: Your threshold ({threshold}) is too extreme! All songs are labeled the same. Try adjusting Step 3.", "warning")
        return redirect(url_for('step3_mining'))

    # 2. Train Model (Logistic Regression - Slide 44)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression()
    
    try:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # 3. Metrics
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # IMPROVEMENT 1: Robust Confusion Matrix
        # We force labels=[0,1] so the matrix is always 2x2, even if predictions are only 0s.
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        
        fig_cm = px.imshow(cm, text_auto=True, 
                           x=['Predicted: Miss', 'Predicted: Hit'], 
                           y=['Actual: Miss', 'Actual: Hit'],
                           labels=dict(x="AI Prediction", y="Real Outcome", color="Count"),
                           color_continuous_scale="Blues", 
                           title="Confusion Matrix (Slide 41)")
        
        # IMPROVEMENT 2: Enhanced Feature Importance
        # We sort by 'Importance' so the biggest factors are at the top
        coefs = pd.DataFrame({'Factor': feature_cols, 'Impact': model.coef_[0]})
        coefs = coefs.sort_values(by='Impact', key=abs, ascending=True) # Sort by magnitude
        
        fig_feat = px.bar(coefs, x='Impact', y='Factor', orientation='h', 
                          title="Success Factors: What drives a Hit?", 
                          color='Impact', 
                          color_continuous_scale='RdBu',
                          labels={'Impact': 'Influence on Popularity (+ means helps, - means hurts)'})

        return render_template('step4_results.html', 
                               acc=f"{report['accuracy']:.2f}",
                               prec=f"{report['1']['precision']:.2f}",
                               rec=f"{report['1']['recall']:.2f}",
                               f1=f"{report['1']['f1-score']:.2f}",
                               cm_json=json.dumps(fig_cm, cls=plotly.utils.PlotlyJSONEncoder),
                               feat_json=json.dumps(fig_feat, cls=plotly.utils.PlotlyJSONEncoder))
                               
    except Exception as e:
        flash(f"Model Training Error: {str(e)}. Try selecting fewer or different features.", "danger")
        return redirect(url_for('step3_mining'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
