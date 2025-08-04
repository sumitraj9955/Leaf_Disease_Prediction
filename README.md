# 🌿 Plant Disease Recognition System

A **Deep Learning + AI-powered** web application to identify plant diseases from leaf images and provide detailed information including **symptoms** and **prevention tips**.

![Demo](Leaf%20Image.jpeg)

---

## 🚀 Features
- **Image Upload** – Upload a leaf image for instant analysis.
- **Deep Learning Prediction** – Detects plant diseases using a trained TensorFlow/Keras model.
- **AI-Powered Insights** – Uses OpenAI GPT to provide short, clear disease descriptions.
- **Local Caching** – Stores frequent disease information to reduce API usage.
- **Streamlit Web App** – Simple and interactive user interface.

---

## 🛠️ Tech Stack
- **Frontend:** [Streamlit](https://streamlit.io/)
- **Backend / Model:** TensorFlow / Keras
- **AI API:** OpenAI GPT-4o-mini
- **Language:** Python 3.13
- **Others:** NumPy, Pillow, tqdm

---

## 📂 Project Structure
📦 Leaf_Disease_Prediction
┣ 📜 main.py # Main Streamlit app
┣ 📜 disease_info.py # OpenAI integration & local descriptions
┣ 📜 trained_model.keras # Pre-trained model
┣ 📜 Leaf Image.jpeg # Home page display image
┣ 📜 dataset-Image.png # Dataset illustration
┣ 📜 requirements.txt # Python dependencies
┗ 📜 README.md # Project documentation

yaml
Copy
Edit

---

## 📦 Installation

1️⃣ **Clone the repository**
```bash
git clone https://github.com/your-username/Leaf_Disease_Prediction.git
cd Leaf_Disease_Prediction
2️⃣ Create a virtual environment (optional but recommended)

bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
3️⃣ Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Set OpenAI API Key

Get your API key from OpenAI.

Set it as an environment variable:

bash
Copy
Edit
# Windows (PowerShell)
setx OPENAI_API_KEY "your_api_key_here"

# Linux/Mac
export OPENAI_API_KEY="your_api_key_here"
▶️ Running the App
bash
Copy
Edit
streamlit run main.py
Then open the given local URL (e.g., http://localhost:8501) in your browser.

📊 Dataset Info
Source: PlantVillage Dataset

Classes: 38 classes (healthy + diseased leaves)

Images: ~87K RGB images

Split: 80% Training, 20% Validation, plus a separate test set.

📷 How It Works
Upload Image → Leaf photo uploaded by user.

Model Prediction → TensorFlow model predicts disease class.

AI Description → OpenAI API generates details on symptoms & prevention.

Display Results → Shows class name & detailed description.

📌 Example Output
Prediction:

vbnet
Copy
Edit
Model is predicting it's: Tomato Early blight
Description:

Early blight on tomatoes causes dark, concentric rings on older leaves, leading to yellowing and leaf drop.
Common in warm, humid conditions. Prevent with crop rotation, resistant varieties, and timely fungicide use.

🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

📜 License
This project is licensed under the MIT License.

🙌 Acknowledgements
PlantVillage Dataset

Streamlit

OpenAI

TensorFlow/Keras community
