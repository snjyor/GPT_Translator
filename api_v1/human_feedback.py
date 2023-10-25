from flask_restful import Resource
from flask import request
from config.config import cfg
from modules.human_feedback import HumanFeedbackModule


class HumanFeedback(Resource):
    def post(self):
        try:
            need_translate = request.json.get("need_translate")
            translation = request.json.get("translation")
            source_lang = request.json.get("source_lang")
            target_lang = request.json.get("target_lang")
            assert type(need_translate) == type(translation), "need_translate and translation should be the same type"
            assert source_lang in cfg.SUPPORTED_LANGUAGES, f"source_lang should be in {cfg.SUPPORTED_LANGUAGES}"
            assert target_lang in cfg.SUPPORTED_LANGUAGES, f"target_lang should be in {cfg.SUPPORTED_LANGUAGES}"
            human_client = HumanFeedbackModule()
            whole_es_data = human_client.save_feedback(need_translate, translation, source_lang, target_lang)
            result = {
                "code": 200,
                "message": "save success",
                "data": {
                    "insert_uid": [item["_source"].get("uid") for item in whole_es_data]
                }
            }
        except Exception as e:
            result = {
                "code": 500,
                "message": f"Human feedback went wrong, DETAIL: ```{e}```",
                "data": {}
            }
        return result