from api_v1.ai_translator import AITranslator, HumanFeedback, SupportLanguages


def register_api_v1(api):
    api.add_resource(AITranslator, "/v1/ai_translate/translate")
    api.add_resource(HumanFeedback, "/v1/ai_translate/feedback")
    api.add_resource(SupportLanguages, "/v1/ai_translate/languages")

