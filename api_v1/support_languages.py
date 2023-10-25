from flask_restful import Resource
from config.config import cfg


class SupportLanguages(Resource):
    def get(self):
        result = {
            "code": 200,
            "message": "success",
            "data": {
                "supported_languages": cfg.SUPPORTED_LANGUAGES
            }
        }
        return result
