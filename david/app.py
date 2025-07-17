from flask import Flask, request, Response, render_template_string
import os
from io import BytesIO
from gtts import gTTS
import base64
from gtts.lang import tts_langs
import logging


# Flask App Settings
app = Flask(__name__)

# Log Settings
file_handler = logging.FileHandler("input_log.txt", encoding="utf-8")
file_handler.setLevel(logging.INFO)

# Log Format: [time] - [Log Level] - [Log Message]
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Add Log Handler to Flask App
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)


# gTTS available langs
SUPPORTED_LANGUAGES = tts_langs()

# Default Language Setting
DEFAULT_LANG = os.getenv("DEFAULT_LANG", "ko")

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text to Speech</title>
</head>
<body>
    <h1>Text to Speech Advanced</h1>
    <form method="POST">
      <label for="input_text">이름 또는 문장을 입력하세요:</label>
      <input type="text" id="input_text" name="input_text" value="{{ input_text or '' }}" required>

      <label for="lang">언어 선택:</label>
      <select id="lang" name="lang">
        {% for code, name in supported_langs.items() %}
            <option value="{{ code }}" {% if code == selected_lang %}selected{% endif %}>{{ name }} ({{ code }})</option>
        {% endfor %}
      </select>

      <button type="submit">음성 듣기</button>
    </form>

    {% if error %}
      <p class="error">오류: {{ error }}</p>
    {% endif %}

    {% if audio %}
      <audio controls autoplay>
        <source src="data:audio/mpeg;base64,{{ audio }}">
      </audio>
      <div class="download-container">
        <a href="data:audio/mpeg;base64,{{ audio }}" download="tts_output.mp3" class="download-link">
        MP3로 저장하기 📥
        </a>
      </div>
    {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    # text = "Hello, DevOps"

    # Basic context to deliver for template
    context = {"supported_langs": SUPPORTED_LANGUAGES, "selected_lang": DEFAULT_LANG}

    if request.method == "POST":
        text = request.form.get("input_text")
        lang = request.form.get("lang", DEFAULT_LANG)

        # 컨텍스트 업데이트
        context["input_text"] = text
        context["selected_lang"] = lang

        # Validation for available langs
        if lang not in SUPPORTED_LANGUAGES:
            context["error"] = "지원되지 않는 언어입니다."
            return render_template_string(HTML_TEMPLATE, **context)

        # Exception Handling: Empty text case
        if not text or not text.strip():
            return render_template_string(
                HTML_TEMPLATE,
                error="음성으로 변환할 텍스트를 입력해주세요.",
                selected_lang=lang,
            )

        try:
            app.logger.info(f"Lang: {lang}, Text: {text}")

            fp = BytesIO()
            tts = gTTS(text=text, lang=lang, tld="com")
            tts.write_to_fp(fp)
            fp.seek(0)
            audio_b64 = base64.b64encode(fp.read()).decode("utf-8")
            context["audio"] = audio_b64
        except Exception as e:
            print(f"An error occurred: {e}")
            app.logger.error(f"gTTS Error for input '{text}': {e}")

            context["error"] = (
                "음성 변환에 실패했습니다. 유효하지 않은 언어이거나 네트워크 문제일 수 있습니다."
            )

    return render_template_string(HTML_TEMPLATE, **context)


if __name__ == "__main__":
    app.run("0.0.0.0", 80)
