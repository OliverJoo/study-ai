from flask import Flask, request, Response, render_template
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


@app.route("/", methods=["GET", "POST"])
def home():
    # text = "Hello, DevOps"

    # Basic context to deliver for template
    context = {"supported_langs": SUPPORTED_LANGUAGES, "selected_lang": DEFAULT_LANG}

    if request.method == "POST":
        text = request.form.get("input_text")
        lang = request.form.get("lang", DEFAULT_LANG)

        # context update but left input data
        context["input_text"] = text
        context["selected_lang"] = lang

        # Validation for available langs
        if lang not in SUPPORTED_LANGUAGES:
            context["error"] = "지원되지 않는 언어입니다."
        elif not text or not text.strip():
            context["error"] = "음성으로 변환할 텍스트를 입력해주세요."
        else:
            try:
                app.logger.info(f"Lang: {lang}, Text: {text}")

                fp = BytesIO()
                tts = gTTS(text=text, lang=lang, tld="com")
                tts.write_to_fp(fp)
                fp.seek(0)
                audio_b64 = base64.b64encode(fp.read()).decode("utf-8")
                context["audio"] = audio_b64
            except Exception as e:
                # print(f"An error occurred: {e}")
                app.logger.error(f"gTTS Error for input '{text}': {e}")
                context["error"] = (
                    "음성 변환에 실패했습니다. 유효하지 않은 언어이거나 네트워크 문제일 수 있습니다."
                )
                

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run("0.0.0.0", 80)
