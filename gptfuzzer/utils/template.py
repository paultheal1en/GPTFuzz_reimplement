QUESTION_PLACEHOLDER = '[INSERT PROMPT HERE]'
"""
Biến toàn cục định nghĩa chuỗi đại diện cho vị trí chèn câu hỏi trong template.
"""


def synthesis_message(question, prompt):
    """
    Tổng hợp tin nhắn bằng cách chèn câu hỏi vào template.
    
    Args:
        question: Câu hỏi cần chèn vào template
        prompt: Template chứa placeholder
        
    Returns:
        Tin nhắn đã được tổng hợp hoặc None nếu không tìm thấy placeholder
    """
    if QUESTION_PLACEHOLDER not in prompt:
        return None

    return prompt.replace(QUESTION_PLACEHOLDER, question)
