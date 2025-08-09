import os
from flask import Flask, request, render_template, session, jsonify
from google import genai
from google.genai import types
from dotenv import load_dotenv
import secrets

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # For session management

# Setup Gemini client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

model = "gemini-2.5-pro"

# Route for home page
@app.route("/")
def index():
    # Initialize chat history if not exists
    if 'chat_history' not in session:
        session['chat_history'] = []
        # Add initial bot greeting
        session['chat_history'].append({
            'type': 'bot',
            'message': "Hello there! I'm Lex, your dental expert. How can I help you today with your dental problems? Please describe your symptoms in detail so I can assist you better."
        })
    
    return render_template("index.html", chat_history=session['chat_history'])

# API route for sending messages
@app.route("/send_message", methods=["POST"])
def send_message():
    try:
        user_input = request.json.get("message", "").strip()
        if 'chat_history' not in session:
            session['chat_history'] = []

        # Add user message to chat history
        session['chat_history'].append({
            'type': 'user',
            'message': user_input
        })

        # Build conversation contents for Gemini
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text="""Your name is Lex and you are a professional dental expert.
Your role is to assist patients with their dental problems with care and professionalism.
You are not allowed to answer any other questions except about dental-related things.

IMPORTANT GUIDELINES:
1. Be formal, polite, and show genuine concern for the patient's wellbeing
2. Ask detailed questions about their symptoms, pain level, duration, location, etc.
3. Provide helpful suggestions and remedies when appropriate
4. After giving advice, always follow up by asking "Did this help with your issue?" or "How are you feeling now?"
5. If your suggestions don't work or the issue seems serious, recommend scheduling a consultation: "I recommend scheduling a consultation with a dental professional for a proper examination. Click here to book an appointment: https://calendly.com/gcloud1241/30min"
6. Only provide the Calendly link when your suggestions haven't helped or for serious issues
7. Keep responses professional but caring, around 2-3 sentences
8. Always prioritize patient safety and recommend professional care when needed
9. When providing the appointment link, always use the exact format: "Click here to book an appointment: https://calendly.com/gcloud1241/30min"
10. If you need to provide multiple suggestions, remedies, or pieces of information, always format them as bullet points for clarity.
"""),
                ],
            ),
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(text="""Hello there! I'm Lex, your dental expert. How can I help you today with your dental problems? Please describe your symptoms in detail so I can assist you better."""),
                ],
            )
        ]
        
        # Add recent conversation history (last 10 messages for context)
        recent_history = session['chat_history'][-11:-1]  # Exclude the just-added user message
        for msg in recent_history:
            if msg['type'] == 'user':
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=msg['message'])]
                ))
            elif msg['type'] == 'bot':
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=msg['message'])]
                ))
        
        # Add current user message
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_input)]
        ))

        tools = [
            types.Tool(googleSearch=types.GoogleSearch()),
        ]

        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
            tools=tools,
        )

        # Stream Gemini response
        full_response = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.text:
                full_response += chunk.text

        # Add bot response to chat history
        session['chat_history'].append({
            'type': 'bot',
            'message': full_response
        })
        
        # Keep only last 50 messages to prevent session from getting too large
        if len(session['chat_history']) > 50:
            session['chat_history'] = session['chat_history'][-50:]
        
        session.modified = True
        
        return jsonify({
            "success": True,
            "bot_response": full_response
        })
        
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Route to clear chat history
@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    session['chat_history'] = []
    # Add initial bot greeting
    session['chat_history'].append({
        'type': 'bot',
        'message': "Hello there! I'm Lex, your dental expert. How can I help you today with your dental problems? ."
    })
    session.modified = True
    return jsonify({"success": True})


# Run the app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
