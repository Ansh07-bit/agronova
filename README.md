# AgroNova

AgroNova is an AI-powered agricultural assistance platform focused on **Arecanut cultivation and crop health**.

The system combines machine learning models with a generative AI assistant to provide farmers with useful information about arecanut price prediction, soil analysis, disease detection, and agricultural guidance.

## Features

### 1. Arecanut Price Prediction

Predicts the estimated arecanut price based on:

* District
* Variety
* Date

The system also generates a price trend covering three months before and three months after the selected date.

### 2. Soil Analysis

Analyzes an uploaded soil image and predicts the soil type.

Supported soil classes:

* Laterite Soil
* Alluvial Soil
* Red Soil
* Black Soil

The system also provides an arecanut suitability score.

### 3. Disease Detection

AgroNova supports multiple image-based detection models:

* Leaf disease detection
* Trunk disease detection
* Crop disease detection
* Arecanut nut quality classification

### 4. AI Agricultural Assistant

The AgroNova expert chatbot uses Gemini to answer questions related to:

* Arecanut cultivation
* Soil
* Diseases
* Fertilizers
* Irrigation
* Pests
* Weather and climate
* Crop management

## Machine Learning Models

The trained models are hosted separately on Hugging Face.

**Hugging Face Repository:**

`Ansh2005A/agronova_models`

Models include:

* `xgb_price_model.json`
* `soil_type_resnet50_finetuned.h5`
* `arecanut_quality_effnet_initial.h5`
* `leaf_disease_resnet50_finetuned.h5`
* `trunk_disease_resnet50_finetuned.h5`
* `nut_disease_resnet50_finetuned.h5`

The application downloads the required models from Hugging Face instead of storing the large model files inside the GitHub repository.

## Technology Stack

### Backend

* Python
* Flask
* SQLite
* REST APIs

### Machine Learning

* XGBoost
* TensorFlow / Keras
* Scikit-learn
* NumPy
* Pandas

### AI

* Google Gemini
* Hugging Face Hub

### Frontend

* HTML
* CSS
* JavaScript

## Project Structure

```text
AgroNova/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
│   ├── home.html
│   ├── login.html
│   ├── signup.html
│   ├── price_prediction.html
│   ├── soil_analysis.html
│   ├── disease_detection.html
│   └── expert_chatbot.html
│
└── static/
    ├── css/
    ├── js/
    └── images/
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Ansh07-bit/agronova.git
```

Move into the project:

```bash
cd agronova
```

Create a virtual environment:

```bash
python -m venv myenv
```

Activate it on Windows:

```powershell
.\myenv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

The Gemini API key should not be stored directly in the source code.

Set it as an environment variable.

### Windows PowerShell

```powershell
$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

For deployment, configure the API key using the hosting platform's environment-variable settings.

## Running the Application

Start the Flask application:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Model Hosting

The application uses Hugging Face Hub to retrieve the trained models.

Repository:

`Ansh2005A/agronova_models`

This keeps the GitHub repository focused on source code while the larger ML model files are hosted separately.

## Deployment

The planned deployment architecture is:

```text
GitHub
   │
   │ Source Code
   ▼
Vercel
   │
   ├── Flask Application
   │
   └── Environment Variables
           │
           ▼
       Gemini API

Hugging Face
   │
   └── Trained ML Models
```

## Security

Do not commit the following files or information to GitHub:

* API keys
* `.env` files
* `users.db`
* Virtual environments
* Hugging Face access tokens
* Other private credentials

These should be excluded using `.gitignore` and configured through environment variables when required.

## Project Status

AgroNova is currently under development and is being prepared for cloud deployment.

## Author

**Ansh**

GitHub: 

```
https://github.com/Ansh07-bit
```
