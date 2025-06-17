from flask import Flask, render_template, request
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('career_model.h5')

# Mappings to match the training data
interest_map = {
    'ai': 0,
    'web_dev': 3,
    'cybersecurity': 1,
    'data_science': 2  # Add if needed
}

personality_map = {
    'introvert': 0,
    'extrovert': 1,
    'ambivert': 2
}

career_labels = ['AI Engineer', 'Data Scientist', 'Frontend Developer', 'Security Analyst']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Step 1: Read skills from checkboxes
    selected_skills = request.form.getlist('skills')
    java = 3 if 'java' in selected_skills else 2 if 'python' in selected_skills else 1
    python = 3 if 'python' in selected_skills else 2 if 'java' in selected_skills else 1
    c = 3 if 'c++' in selected_skills else 2 if 'networking' in selected_skills else 1

    # Step 2: Read interest (take first match from known ones)
    selected_interests = request.form.getlist('interests')
    interest = next((i for i in selected_interests if i in interest_map), 'ai')
    interest_encoded = interest_map[interest]

    # Step 3: Read and encode personality
    personality = request.form.get('personality', 'introvert')
    personality_encoded = personality_map.get(personality, 0)

    # Step 4: Prepare input
    input_data = np.array([[java, python, c, interest_encoded, personality_encoded]])
    input_data = input_data.reshape((1, 5, 1))

    # Step 5: Predict top 2 careers
    prediction = model.predict(input_data)
    top_indices = prediction[0].argsort()[-2:][::-1]
    recommended = [career_labels[i] for i in top_indices]

    return render_template('result.html', careers=recommended)

if __name__ == '__main__':
    app.run(debug=True)
