import hashlib
import re
from regex import regex
import openai
import time
from elasticsearch7 import Elasticsearch, helpers
import tiktoken

from config.config import cfg


def token_usage(text):
    """
    计算token使用量
    :param text: str, 输入文本
    :return: token使用量
    """
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))


def get_entities(sentence):
    """
    获取实体
    :param sentence: str, 输入文本
    :return: 实体列表
    """
    doc = cfg.NLP(sentence)
    entities = [ent.text for sent in doc.sentences for ent in sent.ents]
    return list(set([item.lower() for item in entities]))


def get_entities_by_nltk(sentence):
    """
    获取实体
    :param sentence: str, 输入文本,使用nltk简单分,速度更快
    :return: 实体列表
    """
    doc = cfg.NLTK(sentence)
    entities = [x for x, tag in doc if tag in ["NN", "NNP"]]
    return list(set([item.lower() for item in entities]))


def embedding(sentence):
    """
    获取文本的embedding
    :param sentence: str/list, 输入文本
    :return: embedding列表
    """
    if not sentence:
        return []
    response = openai.Embedding.create(
        input=sentence,
        engine="text-embedding-ada-002"
    )
    embeddings = [dict(item).get('embedding') for item in response['data']]
    return embeddings


def token_usage_from_messages(messages, model="gpt35"):
    """
    返回消息列表使用的令牌数。
    """
    mapping = {
        "gpt35": "gpt-3.5-turbo-0301",
        "gpt4-8k": "gpt-4-0314",
        "gpt4-32k": "gpt-4-0314",
    }
    encoding = tiktoken.encoding_for_model(mapping.get(model, model))
    if real_model := mapping.get(model):
        return token_usage_from_messages(messages, real_model)
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def gpt_request(message, translated_result=""):
    """
    gpt3请求
    :param message: list, 输入message
    :param translated_result: str, 翻译结果
    :return: str, 回复文本
    """
    if cfg.USE_AZURE_AI:
        response = openai.ChatCompletion.create(
            engine=cfg.AZURE_GPT_ENGINE,
            messages=message,
            temperature=0.5,  # 值在[0,1]之间，越大表示回复越具有不确定性
            max_tokens=cfg.MAX_TOKENS,  # 回复最大的字符数
            frequency_penalty=0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容
            presence_penalty=0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容
        )
    else:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=message,
            temperature=0.5,  # 值在[0,1]之间，越大表示回复越具有不确定性
            max_tokens=cfg.MAX_TOKENS,  # 回复最大的字符数
            frequency_penalty=0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容
            presence_penalty=0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容
        )
    content = response['choices'][0].get("message").get("content")
    translated_result = f"{translated_result}{content}"

    while response['choices'][0].get("finish_reason") != "stop":
        message.append({"role": "assistant", "content": content})
        message.append({"role": "user", "content": "Well translated, but the output does not end, please continue the output."})
        delete_oldest_history_message(message)
        translated_result = gpt_request(message, translated_result)
        return translated_result
    return translated_result


def delete_oldest_history_message(message):
    """
    如果token不够，删除最久远的历史消息
    """
    message_tokens = token_usage_from_messages(message, cfg.AZURE_GPT_ENGINE)
    while message_tokens >= cfg.MAX_TOKENS:
        # 把最久远的上文丢掉, 索引0, 1是init的初始prompt, 索引2是最久远的message
        message.pop(2)
        # 重新计算token, 如果token不够，继续丢
        message_tokens = token_usage_from_messages(message, cfg.AZURE_GPT_ENGINE)


def json_regex(text):
    """
    从文本中提取json字符串
    :param text: str, 输入文本
    :return: 可json.loads()的json字符串
    """
    try:
        json_pattern = regex.compile(r"\{(?:[^{}]|(?R))*\}")
        json_match = json_pattern.search(text)
        json_string = json_match.group(0)
    except Exception as err:
        print(f"json_regex error: {err}")
        json_string = ""
    return json_string


def md5_hash(content):
    """
    计算md5
    :param content: 输入文本
    :return: md5
    """
    md5 = hashlib.md5()
    if isinstance(content, str):
        content = content.encode('utf-8')
    md5.update(content)
    return md5.hexdigest()


def format_es_data(source, doc_id):
    """
    格式化es数据
    :param source: dict, 输入数据
    :param doc_id: str, 文档id
    :return: dict, es数据
    """
    return {
        "_index": cfg.INDEX,
        "_id": doc_id,
        "_source": source,
        "_type": "_doc"
    }


class Elastic:
    def __init__(self, index):
        self.es = Elasticsearch(
            cfg.ELASTIC_SERVER, http_auth=(cfg.ELASTIC_USERNAME, cfg.ELASTIC_PASSWORD)
        )
        self.index_name = index

    def es_search(self, should_query, top_k=3):
        if not should_query:
            return []
        response = self.es.search(
            index=self.index_name,
            query={
                "bool": {
                    "should": should_query,
                    "minimum_should_match": 1
                }
            },
            size=top_k
        )
        result = response["hits"]["hits"]
        return [item["_source"] for item in result]

    def insert_into_es(self, row_list, batch=False):
        try_num = cfg.DATA_INSERT_TRY_NUM
        if batch:
            while True:
                try:
                    helpers.bulk(self.es, row_list)
                    break
                except Exception as err:
                    print(f"insert data went wrong! detail: {err}")
                    time.sleep(5)
                    try_num -= 1
                    if try_num == 0:
                        break
        else:
            for content in row_list:
                while True:
                    try:
                        self.es.index(index=self.index_name, document=content.get("_source"), id=content.get("_id"))
                        break
                    except Exception as err:
                        print(f"insert data went wrong! detail: {err}")
                        time.sleep(2)
                        try_num -= 1
                        if try_num == 0:
                            break
                        continue

def force_breakdown(txt, limit, get_token_fn):
    """
    当无法用标点、空行分割时，我们用最暴力的方法切割
    """
    for i in reversed(range(len(txt))):
        if get_token_fn(txt[:i]) < limit:
            return txt[:i], txt[i:]
    return "Tiktoken未知错误", "Tiktoken未知错误"


def cut(txt_tocut, must_break_at_empty_line, limit, break_anyway=False):
    """
    递归切分文本
    :txt_tocut:待切分文本
    :must_break_at_empty_line:是否切分空行
    :limit: token限制
    :break_anyway:是否使用暴力分割
    :return: 切分后文本列表
    """
    if token_usage(txt_tocut) <= limit:
        return [txt_tocut]
    else:
        lines = txt_tocut.split('\n')
        print(lines)
        estimated_line_cut = limit / token_usage(txt_tocut) * len(lines)
        estimated_line_cut = int(estimated_line_cut)
        cnt = 0
        for cnt in reversed(range(estimated_line_cut)):
            if must_break_at_empty_line:
                if lines[cnt] != "":
                    continue
            prev = "\n".join(lines[:cnt])
            post = "\n".join(lines[cnt:])
            if token_usage(prev) < limit:
                break
        if cnt == 0:
            if break_anyway:
                prev, post = force_breakdown(txt_tocut, limit, token_usage)
            else:
                raise RuntimeError(f"存在一行极长的文本！{txt_tocut}")
        result = [prev]
        result.extend(cut(post, must_break_at_empty_line, limit, break_anyway=break_anyway))
        return result

def breakdown_text(txt, limit):
    """
    用不同的分隔符切分文本
    :txt:待切分文本
    :limit: token限制
    :return: 切分后文本列表
    """
    args_list = [txt, txt, txt.replace('.', '。\n'), txt.replace('。', '。。\n')]
    kwargs_list = [{"must_break_at_empty_line": True,},
                   {"must_break_at_empty_line": False,},
                   {"must_break_at_empty_line": False,},
                   {"must_break_at_empty_line": False}]
    kwargs_list = [{**kw, "limit": limit} for kw in kwargs_list]
    count = 0
    for args, kwargs in zip(args_list, kwargs_list):
        try:
            res = cut(args, **kwargs)
            return res
        except RuntimeError as e:
            count += 1
            if count == 3:
                return [r.replace('。\n', '.').replace('。', '.') for r in cut(args, **kwargs)]
            elif count == 4:
                return [r.replace('。。\n', '。').replace('。。', '。') for r in cut(args, **kwargs)]
            elif count == 5:
                return cut(txt, must_break_at_empty_line=False, limit=limit, break_anyway=True)
            continue


def split_sentences(text):
    """
    使用正则表达式匹配中英文句号进行切割
    :param text: 待切分文本
    :return: 切分后文本列表
    """
    sentences = re.split(r'(?<=\D)[\u3002\.](?=\D|$)', text)
    # 把句号加回去
    sentences = [f"{item}。" if re.search(r"[\u4e00-\u9fa5]", item) else f"{item}." for item in sentences if item.strip()!=""]
    return sentences


def cut_text_as_short_as_possible(text, limit=None):
    """
    将文本切分为尽可能短的句子
    :param text: 待切分文本
    :param limit: token限制
    :return: 切分后文本列表
    """
    cfg.TEXT_TOKEN_LIMIT = limit if limit else cfg.TEXT_TOKEN_LIMIT
    return [text] if token_usage(text) <= cfg.TEXT_TOKEN_LIMIT else split_sentences(text)


if __name__ == '__main__':
    test = "值得注意的是，《意见》还提到，要扎实做好稳地价、稳[房价]、稳预期工作，稳妥有序推进房地产风险化解处置。严格落实地方政府债务限额管理，坚决遏制新增隐性债务。\n\n**“[中特估]”又飙了**\n\n**值得注意的是，今天 中特估这个板块又飙了，可以说对市场起到了较大的支撑作用。**\n\n![]\n\n那么，究竟是何缘故呢？\n\n首先，从财政部的数据来看，1—4月，国有企业营业总收入262281.9亿元，同比增长7.1%。从利润总额来看，1—4月，国有企业利润总额14388.1亿元，同比增长15.1%。Microglia belong to tissue-resident macrophages of the central nervous system (CNS), representing the primary innate immune cells. This cell type constitutes ~7% of non-neuronal cells in the mammalian brain and has a variety of biological roles integral to homeostasis and pathophysiology from the late embryonic to adult brain. Its unique identity that distinguishes its \"glial\" features from tissue-resident macrophages resides in the fact that once entering the CNS, it is perennially exposed to a unique environment following the formation of the blood-brain barrier. Additionally, tissue-resident macrophage progenies derive from various peripheral sites that exhibit hematopoietic potential, and this has resulted in interpretation issues surrounding their origin. "
    result = cut_text_as_short_as_possible(test, limit=100)
    print(result)



