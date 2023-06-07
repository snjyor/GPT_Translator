import ast, sys, os
import time
import json
from flask import request
from flask_restful import Resource

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from utils import utils
from config.config import cfg


class BaseTranslator:
    def __init__(self):
        self.es = utils.Elastic(cfg.INDEX)
        self.message = []
        self.source_lang = "English"
        self.target_lang = "Chinese"
        self.fast_model = False


class AITranslator(Resource, BaseTranslator):
    def post(self):
        """
        人工智能翻译接口
        :return:
        """
        try:
            text = request.json.get("text")
            self.source_lang = request.json.get("source_lang", "English")
            self.target_lang = request.json.get("target_lang", "Chinese")
            cfg.AZURE_GPT_ENGINE = request.json.get("engine", "gpt35")
            is_search_term = request.json.get("is_search_term", 0)
            try:
                cfg.IS_SEARCH_TERM_DATA = bool(int(is_search_term)) if isinstance(is_search_term, str) else bool(is_search_term)
            except Exception as e:
                raise Exception(f"is_search_term must be 0 or 1, DETAIL: {e}")
            assert text, "text is required"
            assert self.source_lang in cfg.SUPPORTED_LANGUAGES, f"source_lang must be one of {cfg.SUPPORTED_LANGUAGES}"
            assert self.target_lang in cfg.SUPPORTED_LANGUAGES, f"target_lang must be one of {cfg.SUPPORTED_LANGUAGES}"
            translated, time_cost = self.translate(text)
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

    def translate(self, query):
        """
        翻译主函数
        :param query: 待翻译的文本
        :return: 翻译结果
        """
        cfg.MAX_TOKENS = cfg.ENGINE_TOKENS_MAPPING.get(cfg.AZURE_GPT_ENGINE, 4096)
        cfg.TEXT_TOKEN_LIMIT = cfg.MAX_TOKENS // 4
        text_list = utils.cut_text_as_short_as_possible(query)
        # 这种情况是句子长度不达限制，没有进行切分
        if len(text_list) == 1:
            # translate directly
            self.construct_init_message()
            self.add_message(f"```{text_list[0]}```", role="user")
            translation = utils.gpt_request(self.message)
            translated_text = self.get_translate_result(translation)
        # 这种情况是句子长度达到限制，进行切分
        else:
            self.construct_init_message()
            translate_text = ""
            translated_text = ""
            while text_list:
                translate_text = f"{translate_text}{text_list.pop(0)}"
                translate_token = utils.token_usage(translate_text)
                if translate_token >= cfg.TEXT_TOKEN_LIMIT:
                    translation_item = self.part_translate(translate_text)
                    translated_text = f"{translated_text}{translation_item}"
                    translate_text = ""
            # 最后一段小于cfg.TEXT_TOKEN_LIMIT长度的文本
            if translate_text:
                translation_item = self.part_translate(translate_text)
                translated_text = f"{translated_text}{translation_item}"

        return translated_text

    def part_translate(self, translate_text):
        """
        在这里进行一块一块文本的翻译
        :param translate_text: str
        :return: 翻译结果
        """
        self.add_message(f"```{translate_text}```", role="user")
        # 删除最久远的历史消息直到小于GPT的token限制
        utils.delete_oldest_history_message(self.message)
        translation = utils.gpt_request(self.message)
        self.add_message(translation, role="assistant")
        translation_item = self.get_translate_result(translation)
        return translation_item

    def get_translate_result(self, translation):
        """
        获取翻译结果, 从GPT返回的答案中匹配json，并获取值，如果值不是预期的字符串，刚继续loads并获取里面的target_lang键对应的值，如果都不成功，则直接返回GPT的答案
        :param translation: GPT返回的答案
        :return: 翻译结果
        """
        json_string = utils.json_regex(translation)
        result = json.loads(json_string).get("result")
        try:
            inner_result = ast.literal_eval(result)
            return inner_result.get(self.target_lang)
        except Exception as e:
            return result

    def construct_init_message(self, reference=None):
        """
        构造GPT请求的message
        :param query: 待翻译的文本
        :param reference: 术语库记忆库匹配结果
        :return: GPT请求的message
        """
        response_format = json.dumps({"result": ""})
        system_message = {"role": "system",
                          "content": f"I want you to act as a translator, spell corrector and improver, you are good at translating any languages to and from each other. Now, I give you a {self.source_lang} sentence, please translate this sentence into {self.target_lang}, and answer with the corrected and improved version. I want you to translate with prettier and more elegant high-level {self.target_lang} words and sentences, but make them more professional. You should only respond in JSON format as described below \nResponse Format: \n ```{response_format}``` \nEnsure the response can be parsed by Python json.loads"}
        self.message.append(system_message)
        if reference:
            reference_message = {"role": "user",
                             "content": f"Here are some standard terminology-translation references that can be used to improve your translation: ```\n{str(reference)}\n```\nPlease translate this sentence into {self.target_lang}: "}
            self.message.append(reference_message)
        else:
            query_message = {"role": "user",
                             "content": f"Please translate this sentence into {self.target_lang}: "}
            self.message.append(query_message)

    def add_message(self, content, role="user"):
        """
        添加message
        :param content: message内容
        :param role: message角色, 默认为user
        """
        self.message.append({"role": role, "content": content})

    def format_should_query(self, should_match, data_type, query_vector):
        """
        格式化should_query
        :param should_match: should_match
        :param data_type: 数据类型
        :param query_vector: 查询向量
        :return: 格式化后的should_query
        """
        if not should_match:
            should_match = [[]]
        if not query_vector:
            return []
        should_query = []
        for index, match_ in enumerate(should_match):
            script_query = {
                "script_score": {
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "match": {
                                        "data_tag": data_type
                                    }
                                }
                            ],
                            "should": match_
                        }
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'source_vector') + 1.0",
                        "params": {
                            "query_vector": query_vector if len(query_vector) == cfg.VECTOR_DIM else query_vector[
                                index]}
                    }
                }
            }
            should_query.append(script_query)
        return should_query

    def entities_search(self, entities, top_k):
        """
        在es中搜索实体信息,不使用向量搜索
        :param entities: entities,实体列表
        :param top_k: top_k,返回条数
        :return: 匹配的实体数据
        """
        should = []
        for item in entities:
            match = {
                "multi_match": {
                    "query": item,
                    "minimum_should_match": "100%"
                }
            }
            should.append(match)
        response = self.es.es_search(should)
        return response


class HumanFeedback(Resource, BaseTranslator):
    def post(self):
        try:
            need_translate = request.json.get("need_translate")
            translation = request.json.get("translation")
            source_lang = request.json.get("source_lang")
            target_lang = request.json.get("target_lang")
            assert type(need_translate) == type(translation), "need_translate and translation should be the same type"
            assert source_lang in cfg.SUPPORTED_LANGUAGES, f"source_lang should be in {cfg.SUPPORTED_LANGUAGES}"
            assert target_lang in cfg.SUPPORTED_LANGUAGES, f"target_lang should be in {cfg.SUPPORTED_LANGUAGES}"
            whole_es_data = []
            source_vector = utils.embedding(need_translate)
            if all([isinstance(need_translate, list), isinstance(translation, list), len(need_translate) == len(translation)]):
                for index, item in enumerate(need_translate):
                    source_data = self.construct_source_data(item, translation[index], source_lang, target_lang, source_vector[index])
                    es_data = utils.format_es_data(source_data, source_data.get("uid"))
                    whole_es_data.append(es_data)
            elif isinstance(need_translate, str):
                source_data = self.construct_source_data(need_translate, translation, source_lang, target_lang, source_vector[0])
                es_data = utils.format_es_data(source_data, source_data.get("uid"))
                whole_es_data.append(es_data)
            self.es.insert_into_es(whole_es_data, batch=True)
            print(f"save success, insert_uid: {[item['_source'].get('uid') for item in whole_es_data]}")
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

    @staticmethod
    def construct_source_data(need_translate, translation, source_lang, target_lang, source_vector, data_tag="memory"):
        """
        构造source_data
        :param need_translate: 需要翻译的文本
        :param translation: 翻译结果
        :param source_lang: 源语言
        :param target_lang: 目标语言
        :param source_vector: 源语言文本向量
        :param data_tag: 数据标签, 默认为memory
        :return: source_data, dict
        """
        source_data = {
            "source": need_translate,
            "target": translation,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "data_tag": data_tag,
            "source_vector": source_vector,
            "uid": utils.md5_hash(f"{need_translate}{source_lang}{target_lang}".lower()),
        }
        return source_data


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
