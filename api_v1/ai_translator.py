import sys, os
from flask import request
from flask_restful import Resource

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from config.config import cfg
from modules.translator import AITranslatorModule


class AITranslator(Resource):
    def post(self):
        """
        人工智能翻译接口
        :return:
        """
        try:
            text = request.json.get("text")
            source_lang = request.json.get("source_lang", "English")
            target_lang = request.json.get("target_lang", "Chinese")
            cfg.AZURE_GPT_ENGINE = request.json.get("engine", "gpt35")
            is_search_term = request.json.get("is_search_term", 0)
            try:
                cfg.IS_SEARCH_TERM_DATA = bool(int(is_search_term)) if isinstance(is_search_term, str) else bool(is_search_term)
            except Exception as e:
                raise Exception(f"is_search_term must be 0 or 1, DETAIL: {e}")
            assert text, "text is required"
            assert source_lang in cfg.SUPPORTED_LANGUAGES, f"source_lang must be one of {cfg.SUPPORTED_LANGUAGES}"
            assert target_lang in cfg.SUPPORTED_LANGUAGES, f"target_lang must be one of {cfg.SUPPORTED_LANGUAGES}"
            translated, time_cost = AITranslatorModule().translate(text, source_lang, target_lang)
            result = {
                "code": 200,
                "message": "success",
                "data": {
                    "text": text,
                    "translated": translated,
                    "source_lang": self.source_lang,
                    "target_lang": self.target_lang,
                }
            }
        except Exception as e:
            result = {
                "code": 500,
                "message": f"Translate went wrong, DETAIL: ```{e}```",
                "data": {}
            }
        return result


if __name__ == '__main__':
    translater = AITranslator()
    import gradio as gr

    client = gr.Interface(
        fn=translater.translate,
        inputs=gr.Textbox(
            lines=6,
            placeholder="Enter Your text here...",
            label="需要翻译的句子"
        ),
        outputs=[
            gr.Textbox(label="翻译结果", lines=6),
            gr.Textbox(label="耗时")
        ]
    )
    client.launch()
