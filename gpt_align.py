import openai
import os

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def align_sentences(chinese_text, vietnamese_text):
    prompt = f"""
    Given the following Chinese and Vietnamese sentences, align them based on meaning:
    
    Chinese: {chinese_text}
    Vietnamese: {vietnamese_text}
    
    Return a JSON list of aligned sentence pairs.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    with open("response.txt", "w", encoding="utf-8") as f:
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                f.write(chunk.choices[0].delta.content)

# Example usage
chinese = "话说天下大势，分久必合，合久必分。周末七国分争，并入于秦。及秦灭之后，楚、汉分争，又并入于汉。汉朝自高祖斩白蛇而起义，一统天下，后来光武中兴，传至献帝，遂分为三国。推其致乱之由，殆始于桓、灵二帝。桓帝禁锢善类，崇信宦官。及桓帝崩，灵帝即位，大将军窦武、太傅陈蕃共相辅佐。时有宦官曹节等弄权，窦武、陈蕃谋诛之，机事不密，反为所害，中涓自此愈横。"
vietnamese = "Thế lớn trong thiên hạ, cứ tan lâu rồi lại hợp, hợp lâu rồi lại tan: Như cuối đời nhà Chu, bảy nước tranh giành xâu xé nhau rồi sau lại hợp về nhà Tần. Đến khi nhà Tần mất, thì Hán Sở tranh hùng rồi sau thiên hạ lại hợp về tay nhà Hán. Nhà Hán, từ lúc vua Cao tổ (Bái Công) chém rắn trắng khởi nghĩa, thống nhất được thiên hạ, sau vua Quang Vũ lên ngôi, rồi truyền mãi đến vua Hiến Đế; lúc bấy giờ lại chia ra thành ba nước. Nguyên nhân gây ra biến loạn ấy là do hai vua Hoàn đế, Linh đế. Vua Hoàn đế tin dùng lũ hoạn quan, cấm cố những người hiền sĩ. Đến lúc vua Hoàn đế băng hà, vua Linh đế lên ngôi nối nghiệp; được quan đại tướng quân Đậu Vũ, quan thái phó Trần Phồn giúp đỡ. Khi ấy, trong triều có bọn hoạn quan là lũ Tào Tiết lộng quyền. Đậu Vũ, Trần Phồn lập mưu định trừ bọn ấy đi, nhưng vì cơ mưu tiết lộ nên lại bị chúng nó giết mất. Từ đấy, bọn hoạn quan ngày càng bạo ngược."

align_sentences(chinese, vietnamese)
