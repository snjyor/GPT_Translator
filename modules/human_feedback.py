from utils import utils
from config.config import cfg


class HumanFeedbackModule:
    def __init__(self):
        self.es = utils.Elastic(cfg.INDEX)

    def save_feedback(self, need_translate: str, translation: str, source_lang: str, target_lang: str):
        """
        :param need_translate: 需要翻译的文本
        :param translation: 翻译结果
        :param source_lang: 源语言
        :param target_lang: 目标语言
        """
        whole_es_data = []
        source_vector = utils.embedding(need_translate)
        if all([isinstance(need_translate, list), isinstance(translation, list),
                len(need_translate) == len(translation)]):
            for index, item in enumerate(need_translate):
                source_data = self.construct_source_data(item, translation[index], source_lang, target_lang,
                                                                 source_vector[index])
                es_data = utils.format_es_data(source_data, source_data.get("uid"))
                whole_es_data.append(es_data)
        elif isinstance(need_translate, str):
            source_data = self.construct_source_data(need_translate, translation, source_lang, target_lang,
                                                     source_vector[0])
            es_data = utils.format_es_data(source_data, source_data.get("uid"))
            whole_es_data.append(es_data)
        self.es.insert_into_es(whole_es_data, batch=True)
        print(f"save success, insert_uid: {[item['_source'].get('uid') for item in whole_es_data]}")
        return whole_es_data

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
