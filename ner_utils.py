import os
import openai
import json

def get_ner_zh(sentence: str):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    openai.api_key = OPENAI_API_KEY

    prompt = f"""
    You are a Classical Chinese Named Entity Recognition (NER) model in historical domain.
    Given the sentence below, identify the named entities and their types.
    THE OUTPUT FORMAT IS JSON.
    BELOW IS AN EXAMPLE:
    Sentence: "玄 德 曰 ： “ 不 如 走 樊 城 以 避 之 。 ”"
    Output: {{"entities":  [["玄", "德"], ["樊", "城"]]}}
    Each entity is a list of characters that form a named entity.
    NO OTHER TEXT, JUST THE JSON OUTPUT.
    Sentence: "{' '.join(sentence)}"
    """

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Try to parse the output
    try:
        response_text = response.choices[0].message.content
        response_dict = json.loads(response_text)
        # Check if 'entities' key exists in the response
        if 'entities' in response_dict:
            return response_dict['entities']
        else:
            raise ValueError("Invalid response format")
    except (ValueError, SyntaxError):
        print("Invalid response format, retrying...")
        return get_ner_zh(sentence)

def get_ner_vn(sentence: str):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    openai.api_key = OPENAI_API_KEY

    prompt = f"""
    You are a Vietnamese Named Entity Recognition (NER) model in historical domain.
    Given the sentence below, identify the named entities and their types.
    THE OUTPUT FORMAT IS JSON.
    BELOW IS AN EXAMPLE:
    Sentence: "Đến khi nhà Tần mất, thì Hán Sở tranh hùng rồi sau thiên hạ lại hợp về tay nhà Hán."
    Output: {{"entities":  [["Tần"], ["Hán", "Sở"], ["Hán"]]}}
    Each entity is a list of words that form a named entity.
    NO OTHER TEXT, JUST THE JSON OUTPUT.
    Sentence: "{sentence}"
    """

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Try to parse the output
    try:
        response_text = response.choices[0].message.content
        response_dict = json.loads(response_text)
        # Check if 'entities' key exists in the response
        if 'entities' in response_dict:
            return response_dict['entities']
        else:
            raise ValueError("Invalid response format")
    except (ValueError, SyntaxError):
        print("Invalid response format, retrying...")
        return get_ner_zh(sentence)

def get_ner_zh_deepseek(sentence: str):
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_API_URL = "https://api.deepseek.com/v1"
    client = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_URL)
    
    prompt = f"""
    You are a Classical Chinese Named Entity Recognition (NER) model in historical domain.
    Given the sentence below, identify the named entities and their types.
    THE OUTPUT FORMAT IS JSON.
    BELOW IS AN EXAMPLE:
    Sentence: "玄 德 曰 ： “ 不 如 走 樊 城 以 避 之 。 ”"
    Output: {{"entities":  [["玄", "德"], ["樊", "城"]]}}
    Each entity is a list of characters that form a named entity.
    NO OTHER TEXT, JUST THE JSON OUTPUT.
    NO ``` MARKDOWN CODE BLOCKS.
    Sentence: "{' '.join(sentence)}"
    """
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    
    # Try to parse the output
    try:
        response_text = response.choices[0].message.content
        response_dict = json.loads(response_text)
        # Check if 'entities' key exists in the response
        if 'entities' in response_dict:
            return response_dict['entities']
        else:
            raise ValueError("Invalid response format")
    except (ValueError, SyntaxError):
        print("Invalid response format, retrying...")
        return get_ner_zh_deepseek(sentence)

# TEST
if __name__ == "__main__":
    import sys
    input_file_path = sys.argv[1]
    for line in open(input_file_path, "r", encoding="utf-8"):
        line = line.strip()
        if line:
            split_line = line.split("\t")
            if len(split_line) != 2:
                print(f"Invalid line format: {line}")
                continue
            zh_part = split_line[0].strip()
            vn_part = split_line[1].strip()
            deepseek_zh = get_ner_zh_deepseek(zh_part)
            zh_ner = get_ner_zh(zh_part)
            vn_ner = get_ner_vn(vn_part)
            print(f"sentence: {line}")
            print(f"deepseek_zh: {deepseek_zh}")
            print(f"zh_ner: {zh_ner}")
            print(f"vn_ner: {vn_ner}")
            print("-" * 20)