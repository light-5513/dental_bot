# 🦷 Dental Bot - Lex (Your Dental Expert)

A professional AI-powered dental chatbot built with Flask and Google's Gemini AI. Lex provides expert dental advice, answers questions about dental health, and helps schedule consultations when needed.

## Features

- 💬 Interactive chat interface with real-time responses
- 🤖 AI-powered dental expert using Google Gemini 2.5 Pro
- 🔍 Google Search integration for accurate, up-to-date information
- 📱 Responsive design for mobile and desktop
- 💾 Session-based chat history
- 🔒 Secure API key management
- 📅 Integrated appointment scheduling via Calendly

## Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/light-5513/dental_bot.git
   cd dental_bot
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Copy `.env.example` to `.env`
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your Gemini API key:
     ```
     GEMINI_API_KEY=your_actual_api_key_here
     ```

## Usage

### Development Mode

Run the application in development mode:

```bash
python app.py
```

The application will be available at `http://localhost:5000`

### Production Deployment

For production deployment, it's recommended to use a production-grade WSGI server:

**Using Gunicorn:**

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

**Using Waitress (Windows-compatible):**

```bash
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

### Environment Variables

Configure the following environment variables in your `.env` file:

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Your Google Gemini API key | Yes |

## Project Structure

```
dental_bot/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Chat interface frontend
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore rules
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── LICENSE               # MIT License
```

## How It Works

1. **User Interface**: The chat interface is built with HTML, CSS, and vanilla JavaScript
2. **Backend**: Flask handles HTTP requests and manages chat sessions
3. **AI Processing**: User messages are sent to Google Gemini AI with dental expert instructions
4. **Context Management**: Recent chat history is maintained for contextual responses
5. **Appointment Scheduling**: When needed, users are directed to a Calendly booking link

## Features in Detail

### Chat Management
- Persistent chat history during session
- Clear chat functionality
- Automatic message history pruning (keeps last 50 messages)

### Error Handling
- Graceful error messages for API failures
- Input validation
- Network error handling

### Security
- Environment-based API key management
- Session-based authentication
- Secure secret key generation

## API Endpoints

- `GET /` - Main chat interface
- `POST /send_message` - Send a message to the bot
- `POST /clear_chat` - Clear chat history

## Configuration

### Modify AI Behavior

Edit the system prompt in `app.py` (lines 58-71) to customize Lex's personality and responses.

### Adjust Session Settings

Modify session configuration in `app.py`:
- Message history limit (line 128)
- Context window size (line 83)

## Troubleshooting

### Common Issues

**API Key Error:**
- Ensure your `GEMINI_API_KEY` is correctly set in the `.env` file
- Verify the API key is valid at [Google AI Studio](https://makersuite.google.com/)

**Dependencies Not Found:**
- Run `pip install -r requirements.txt` to install all dependencies
- Ensure you're using Python 3.8 or higher

**Port Already in Use:**
- Change the port in `app.py` (line 156) or use an environment variable:
  ```python
  app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
  ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google Gemini AI for powering the dental expertise
- Flask framework for the web application
- Calendly for appointment scheduling integration

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**⚠️ Disclaimer:** This chatbot provides general dental information and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your dentist or other qualified healthcare provider with any questions you may have regarding a dental condition.
