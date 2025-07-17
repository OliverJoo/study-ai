from flask import Flask, request, Response, render_template_string
import os
from io import BytesIO
from gtts import gTTS
import base64

# Flask App Settings
app = Flask(__name__)

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
    <h1>Text to Speech</h1>
    <form method="POST">
      <label for="input_text">이름 또는 문장을 입력하세요:</label>
      <input type="text" id="input_text" name="input_text" value="{{ input_text or '' }}" required>
      
      <label for="lang">언어 선택:</label>
      <select id="lang" name="lang">
        <option value="ko" {% if selected_lang == 'ko' %}selected{% endif %}>한국어</option>
        <option value="en" {% if selected_lang == 'en' %}selected{% endif %}>영어</option>
        <option value="ja" {% if selected_lang == 'ja' %}selected{% endif %}>일본어</option>
        <option value="es" {% if selected_lang == 'es' %}selected{% endif %}>스페인어</option>
      </select>
    
      <button type="submit">음성 듣기</button>
    </form>
    {% if error %}
      <p class="error">오류: {{ error }}</p>
    {% endif %}
    
    {% if audio %}
      <audio controls autoplay>
        <source src="data:audio/mpeg;base64,{{ audio }}">
        브라우저가 오디오 태그를 지원하지 않습니다.
      </audio>
    {% endif %}

</body>
</html>
"""


@app.route("/", methods=['GET', 'POST'])
def home():
    # text = "Hello, DevOps"
    if request.method == 'POST':
        text = request.form.get('input_text')
        lang = request.form.get('lang', DEFAULT_LANG)

        # Exception Handling: Empty text case
        if not text or not text.strip():
            return render_template_string(
                HTML_TEMPLATE, 
                error="음성으로 변환할 텍스트를 입력해주세요.", 
                selected_lang=lang
            )
        
        try:
            # Execute TTS Translation
            fp = BytesIO()
            tts = gTTS(text=text, lang=lang, tld='com')
            tts.write_to_fp(fp)
            fp.seek(0)

            # Prep to insert Audio data
            audio_data = base64.b64encode(fp.read()).decode("utf-8")

            # If Success,
            return render_template_string(
                HTML_TEMPLATE, 
                audio=audio_data, 
                input_text = text,
                selected_lang=lang
            )
        except Exception as e:
            print(f"Error: {e}")
            return render_template_string(
                HTML_TEMPLATE,
                error="음성 변환 실패하였습니다. 유효하지 않은 언어 또는 네트워크 문제 일 수 있습니다.",
                input=text,
                selected_lang=lang,
            )

    # lang = request.args.get("lang", DEFAULT_LANG)
    # fp = BytesIO()
    # gTTS(text, "com", lang).write_to_fp(fp)

    return render_template_string(HTML_TEMPLATE, selected_lang=DEFAULT_LANG)


if __name__ == "__main__":
    app.run("0.0.0.0", 80)
