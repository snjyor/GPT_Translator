from config.config import cfg
from utils import utils
import json
import ast


class AITranslatorModule:
    def __init__(self):
        self.es = utils.Elastic(cfg.INDEX)
        self.message = []
        self.source_lang = "English"
        self.target_lang = "Chinese"

    def translate(self, query: str, source_lang: str = "English", target_lang: str = "Chinese"):
        """
        翻译主函数
        :param query: 待翻译的文本
        :param source_lang: 源语言
        :param target_lang: 目标语言
        :return: 翻译结果
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
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
