import os
import cv2
import numpy as np
import json
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = Flask(__name__)
app.secret_key = "super_secret_key"

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# --------------------------
# File Validation
# --------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --------------------------
# Skin Tone Detection
# --------------------------
def detect_skin_tone(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w, _ = image.shape
    center = image[h//4:h//2, w//4:w//2]

    avg_color = np.mean(center, axis=(0, 1))
    r, g, b = avg_color

    brightness = (r + g + b) / 3

    if brightness > 200:
        tone = "Fair"
    elif brightness > 150:
        tone = "Medium"
    elif brightness > 100:
        tone = "Olive"
    else:
        tone = "Deep"

    return tone


# --------------------------
# AI Recommendation
# --------------------------
def get_ai_recommendation(tone, gender):

    prompt = f"""
    User Skin Tone: {tone}
    Gender: {gender}

    Return ONLY valid JSON:

    {{
        "outfit": ["point1", "point2", "point3"],
        "palette": ["Color1", "Color2", "Color3"],
        "accessories": ["point1", "point2"],
        "hairstyle": ["point1"],
        "why": ["point1"]
    }}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=600
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "")
    start = raw.find("{")
    end = raw.rfind("}") + 1
    clean_json = raw[start:end]

    return json.loads(clean_json)


# --------------------------
# Routes
# --------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():

    file = request.files['image']
    gender = request.form.get('gender')

    if file.filename == '' or not allowed_file(file.filename):
        return "Invalid file"

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    tone = detect_skin_tone(filepath)
    recommendation = get_ai_recommendation(tone, gender)

    # Store in session
    session['recommendation'] = recommendation
    session['gender'] = gender
    session['tone'] = tone

    return redirect(url_for('recommendations'))


@app.route('/recommendations')
def recommendations():

    recommendation = session.get('recommendation')
    tone = session.get('tone')

    if not recommendation:
        return redirect(url_for('home'))

    return render_template(
        'recommendations.html',
        rec=recommendation,
        tone=tone
    )


@app.route('/shopping')
def shopping():

    recommendation = session.get('recommendation')
    gender = session.get('gender', "Unisex")

    if not recommendation:
        return redirect(url_for('home'))

    palette = recommendation.get("palette", [])

    products = []

    for color in palette:
        products.append({
            "name": f"{color} {gender} Outfit",
            "amazon": f"https://www.amazon.in/s?k={color}+{gender}+clothing",
            "myntra": f"https://www.myntra.com/{color}-{gender.lower()}",
            "ajio": f"https://www.ajio.com/search/?text={color}%20{gender}",
            "flipkart": f"https://www.flipkart.com/search?q={color}+{gender}+clothing",
            "meesho": f"https://www.meesho.com/search?q={color}+{gender}+clothing"
        })

    return render_template("shopping.html", products=products)


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
