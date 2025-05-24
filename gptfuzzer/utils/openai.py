import openai
import logging


def openai_request(messages, model='gpt-3.5-turbo', temperature=1, top_n=1, max_trials=100):
    """
    Gửi yêu cầu đến API của OpenAI và xử lý kết quả trả về.
    
    Args:
        messages: Danh sách các tin nhắn để gửi đến API
        model: Tên mô hình OpenAI sử dụng (mặc định là 'gpt-3.5-turbo')
        temperature: Độ đa dạng của kết quả (càng cao càng đa dạng)
        top_n: Số lượng kết quả trả về
        max_trials: Số lần thử lại tối đa khi gặp lỗi
        
    Returns:
        Kết quả từ API OpenAI
        
    Raises:
        ValueError: Nếu chưa cài đặt API key
    """
    if openai.api_key is None:
        raise ValueError(
            "You need to set OpenAI API key manually. `opalai.api_key = [your key]`")

    for _ in range(max_trials):
        try:
            results = openai.Completion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                n=top_n,
            )

            assert len(results['choices']) == top_n
            return results
        except Exception as e:
            logging.warning("OpenAI API call failed. Retrying...")
