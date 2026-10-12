import os
from google import genai
from flask import Blueprint, request, render_template, redirect, url_for, session, flash, jsonify
from app.models.detection import Detection
from app.models.user import User

ai_bp = Blueprint('ai', __name__)

@ai_bp.route('/ask/<detection_id>', methods=['GET', 'POST'])
def ask_ai(detection_id):
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
        
    detection = Detection.get_detection(detection_id)
    if not detection or str(detection.get('user_id')) != session['user_id']:
        flash('Unauthorized access', 'error')
        return redirect(url_for('main.dashboard'))
        
    user = User.get_by_id(session['user_id'])
    
    # Configure Gemini
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your_gemini_api_key_here':
        flash('AI Assistant is currently unavailable (Missing API Key).', 'error')
        return redirect(url_for('detection.result', detection_id=detection_id))
        
    if request.method == 'POST':
        user_message = request.form.get('message')
        language = request.form.get('language', 'English')
        
        # Build prompt with RAG context and STRICT boundary
        prompt = f"""
        You are CropCare AI, an expert agricultural assistant.
        
        STRICT BOUNDARY: 
        You MUST ONLY answer questions related to farming, agriculture, crops, plant diseases, fertilizers, and related topics.
        If the user asks anything unrelated to agriculture (like programming, general knowledge, movies, etc.), you MUST politely decline and state that you can only help with agricultural queries.
        
        Context:
        - Farmer Location: {user.get('state', 'Unknown')}, {user.get('district', 'Unknown')}
        - Crop: {detection.get('crop')}
        - Detected Disease: {detection.get('disease')}
        
        The farmer asks: "{user_message}"
        
        Provide a concise, practical, and agricultural-focused answer based on the context.
        IMPORTANT: Your response MUST be entirely in {language}.
        """
        
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model='gemini-3.5-flash',
                contents=prompt,
            )
            reply = response.text
        except Exception as e:
            error_str = str(e)
            if "503" in error_str or "UNAVAILABLE" in error_str:
                reply = "I'm sorry, the AI service is currently experiencing high demand. Please try again in a few moments."
            else:
                reply = "I'm sorry, I couldn't process your request right now. Please try again later."
            print(f"Gemini API Error: {e}")
            
        return jsonify({"reply": reply})
        
    return render_template('ask_ai.html', detection=detection)
