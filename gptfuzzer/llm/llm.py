import torch
from openai import OpenAI
from fastchat.model import load_model, get_conversation_template
import logging
import time
import concurrent.futures
# from vllm import LLM as vllm
# from vllm import SamplingParams
import google.generativeai as palm
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT


class LLM:
    """
    Lớp cơ sở cho tất cả các mô hình ngôn ngữ lớn (LLM).
    Định nghĩa các phương thức chung mà tất cả các LLM cần triển khai.
    """
    def __init__(self):
        """
        Khởi tạo đối tượng LLM với các thuộc tính cơ bản.
        """
        self.model = None
        self.tokenizer = None

    def generate(self, prompt):
        """
        Phương thức trừu tượng để tạo ra phản hồi từ prompt.
        Cần được triển khai bởi các lớp con.
        
        Args:
            prompt: Chuỗi prompt đầu vào
        """
        raise NotImplementedError("LLM must implement generate method.")

    def predict(self, sequences):
        """
        Phương thức trừu tượng để dự đoán kết quả từ chuỗi đầu vào.
        Cần được triển khai bởi các lớp con.
        
        Args:
            sequences: Danh sách các chuỗi đầu vào
        """
        raise NotImplementedError("LLM must implement predict method.")


class LocalLLM(LLM):
    """
    Lớp đại diện cho mô hình ngôn ngữ lớn chạy cục bộ.
    Sử dụng thư viện FastChat để tải và chạy mô hình.
    """
    def __init__(self,
                 model_path,
                 device='cuda',
                 num_gpus=1,
                 max_gpu_memory=None,
                 dtype=torch.float16,
                 load_8bit=False,
                 cpu_offloading=False,
                 revision=None,
                 debug=False,
                 system_message=None
                 ):
        """
        Khởi tạo mô hình LLM cục bộ.
        
        Args:
            model_path: Đường dẫn đến mô hình
            device: Thiết bị để chạy mô hình ('cuda' hoặc 'cpu')
            num_gpus: Số lượng GPU sử dụng
            max_gpu_memory: Bộ nhớ GPU tối đa sử dụng
            dtype: Kiểu dữ liệu của mô hình
            load_8bit: Có tải mô hình ở định dạng 8-bit hay không
            cpu_offloading: Có sử dụng CPU offloading hay không
            revision: Phiên bản của mô hình
            debug: Chế độ gỡ lỗi
            system_message: Thông điệp hệ thống tùy chỉnh
        """
        super().__init__()

        self.model, self.tokenizer = self.create_model(
            model_path,
            device,
            num_gpus,
            max_gpu_memory,
            dtype,
            load_8bit,
            cpu_offloading,
            revision=revision,
            debug=debug,
        )
        self.model_path = model_path

        if system_message is None and 'Llama-2' in model_path:
            # Sửa đổi cho FastChat mới nhất để sử dụng thông điệp hệ thống chính thức của Llama-2
            self.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. " \
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n" \
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
            "If you don't know the answer to a question, please don't share false information."
        else:
            self.system_message = system_message

    @torch.inference_mode()
    def create_model(self, model_path,
                     device='cuda',
                     num_gpus=1,
                     max_gpu_memory=None,
                     dtype=torch.float16,
                     load_8bit=False,
                     cpu_offloading=False,
                     revision=None,
                     debug=False):
        """
        Tạo và tải mô hình từ đường dẫn.
        
        Args:
            model_path: Đường dẫn đến mô hình
            device: Thiết bị để chạy mô hình
            num_gpus: Số lượng GPU sử dụng
            max_gpu_memory: Bộ nhớ GPU tối đa sử dụng
            dtype: Kiểu dữ liệu của mô hình
            load_8bit: Có tải mô hình ở định dạng 8-bit hay không
            cpu_offloading: Có sử dụng CPU offloading hay không
            revision: Phiên bản của mô hình
            debug: Chế độ gỡ lỗi
            
        Returns:
            Tuple (model, tokenizer) chứa mô hình và tokenizer đã tải
        """
        model, tokenizer = load_model(
            model_path,
            device,
            num_gpus,
            max_gpu_memory,
            dtype,
            load_8bit,
            cpu_offloading,
            revision=revision,
            debug=debug,
        )

        return model, tokenizer

    def set_system_message(self, conv_temp):
        """
        Đặt thông điệp hệ thống cho mẫu hội thoại.
        
        Args:
            conv_temp: Mẫu hội thoại cần đặt thông điệp hệ thống
        """
        if self.system_message is not None:
            conv_temp.set_system_message(self.system_message)

    @torch.inference_mode()
    def generate(self, prompt, temperature=0.01, max_tokens=512, repetition_penalty=1.0):
        """
        Tạo phản hồi từ prompt đầu vào.
        
        Args:
            prompt: Chuỗi prompt đầu vào
            temperature: Nhiệt độ để kiểm soát tính ngẫu nhiên (càng thấp càng xác định)
            max_tokens: Số lượng token tối đa trong phản hồi
            repetition_penalty: Hệ số phạt cho việc lặp lại
            
        Returns:
            Chuỗi phản hồi được tạo ra
        """
        conv_temp = get_conversation_template(self.model_path)
        self.set_system_message(conv_temp)

        conv_temp.append_message(conv_temp.roles[0], prompt)
        conv_temp.append_message(conv_temp.roles[1], None)

        prompt_input = conv_temp.get_prompt()
        input_ids = self.tokenizer([prompt_input]).input_ids
        output_ids = self.model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=False,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_tokens
        )

        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]):]

        return self.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )

    @torch.inference_mode()
    def generate_batch(self, prompts, temperature=0.01, max_tokens=512, repetition_penalty=1.0, batch_size=16):
        """
        Tạo phản hồi cho một loạt các prompt đầu vào.
        
        Args:
            prompts: Danh sách các prompt đầu vào
            temperature: Nhiệt độ để kiểm soát tính ngẫu nhiên
            max_tokens: Số lượng token tối đa trong mỗi phản hồi
            repetition_penalty: Hệ số phạt cho việc lặp lại
            batch_size: Kích thước lô để xử lý
            
        Returns:
            Danh sách các phản hồi tương ứng với các prompt
        """
        prompt_inputs = []
        for prompt in prompts:
            conv_temp = get_conversation_template(self.model_path)
            self.set_system_message(conv_temp)

            conv_temp.append_message(conv_temp.roles[0], prompt)
            conv_temp.append_message(conv_temp.roles[1], None)

            prompt_input = conv_temp.get_prompt()
            prompt_inputs.append(prompt_input)

        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        input_ids = self.tokenizer(prompt_inputs, padding=True).input_ids
        # Tải input_ids theo lô để tránh hết bộ nhớ (OOM)
        outputs = []
        for i in range(0, len(input_ids), batch_size):
            output_ids = self.model.generate(
                torch.as_tensor(input_ids[i:i+batch_size]).cuda(),
                do_sample=False,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_tokens,
            )
            output_ids = output_ids[:, len(input_ids[0]):]
            outputs.extend(self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True, spaces_between_special_tokens=False))
        return outputs


class LocalVLLM(LLM):
    """
    Lớp đại diện cho mô hình ngôn ngữ lớn chạy cục bộ sử dụng thư viện vLLM.
    vLLM cung cấp hiệu suất cao hơn cho việc suy luận với mô hình LLM.
    """
    def __init__(self,
                 model_path,
                 gpu_memory_utilization=0.95,
                 system_message=None
                 ):
        """
        Khởi tạo mô hình vLLM cục bộ.
        
        Args:
            model_path: Đường dẫn đến mô hình
            gpu_memory_utilization: Tỷ lệ sử dụng bộ nhớ GPU
            system_message: Thông điệp hệ thống tùy chỉnh
        """
        super().__init__()
        self.model_path = model_path
        self.model = vllm(
            self.model_path, gpu_memory_utilization=gpu_memory_utilization)
        
        if system_message is None and 'Llama-2' in model_path:
            # Sửa đổi cho FastChat mới nhất để sử dụng thông điệp hệ thống chính thức của Llama-2
            self.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. " \
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n" \
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
            "If you don't know the answer to a question, please don't share false information."
        else:
            self.system_message = system_message

    def set_system_message(self, conv_temp):
        """
        Đặt thông điệp hệ thống cho mẫu hội thoại.
        
        Args:
            conv_temp: Mẫu hội thoại cần đặt thông điệp hệ thống
        """
        if self.system_message is not None:
            conv_temp.set_system_message(self.system_message)

    def generate(self, prompt, temperature=0, max_tokens=512):
        """
        Tạo phản hồi từ prompt đầu vào sử dụng vLLM.
        
        Args:
            prompt: Chuỗi prompt đầu vào
            temperature: Nhiệt độ để kiểm soát tính ngẫu nhiên
            max_tokens: Số lượng token tối đa trong phản hồi
            
        Returns:
            Chuỗi phản hồi được tạo ra
        """
        prompts = [prompt]
        return self.generate_batch(prompts, temperature, max_tokens)

    def generate_batch(self, prompts, temperature=0, max_tokens=512):
        """
        Tạo phản hồi cho một loạt các prompt đầu vào sử dụng vLLM.
        
        Args:
            prompts: Danh sách các prompt đầu vào
            temperature: Nhiệt độ để kiểm soát tính ngẫu nhiên
            max_tokens: Số lượng token tối đa trong mỗi phản hồi
            
        Returns:
            Danh sách các phản hồi tương ứng với các prompt
        """
        prompt_inputs = []
        for prompt in prompts:
            conv_temp = get_conversation_template(self.model_path)
            self.set_system_message(conv_temp)

            conv_temp.append_message(conv_temp.roles[0], prompt)
            conv_temp.append_message(conv_temp.roles[1], None)

            prompt_input = conv_temp.get_prompt()
            prompt_inputs.append(prompt_input)

        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        results = self.model.generate(
            prompt_inputs, sampling_params, use_tqdm=False)
        outputs = []
        for result in results:
            outputs.append(result.outputs[0].text)
        return outputs


class BardLLM(LLM):
    """
    Lớp đại diện cho mô hình Bard của Google.
    Hiện tại chỉ có phương thức generate trống.
    """
    def generate(self, prompt):
        """
        Phương thức chưa được triển khai cho Bard.
        
        Args:
            prompt: Chuỗi prompt đầu vào
        """
        return

class PaLM2LLM(LLM):
    """
    Lớp đại diện cho mô hình PaLM 2 của Google.
    Sử dụng API PaLM để tạo phản hồi.
    """
    def __init__(self,
                 model_path='chat-bison-001',
                 api_key=None,
                 system_message=None
                ):
        """
        Khởi tạo mô hình PaLM 2.
        
        Args:
            model_path: Tên mô hình PaLM 2 (mặc định là 'chat-bison-001')
            api_key: Khóa API để truy cập PaLM 2
            system_message: Thông điệp hệ thống tùy chỉnh
        """
        super().__init__()
        
        if len(api_key) != 39:
            raise ValueError('invalid PaLM2 API key')
        
        palm.configure(api_key=api_key)
        available_models = [m for m in palm.list_models()]
        for model in available_models:
            if model.name == model_path:
                self.model_path = model
                break
        self.system_message = system_message
        # PaLM-2 có giới hạn lớn về số lượng token đầu vào, vì vậy tôi sẽ phát hành các prompt jailbreak ngắn sau này
        
    def generate(self, prompt, temperature=0, n=1, max_trials=1, failure_sleep_time=1):
        """
        Tạo phản hồi từ prompt đầu vào sử dụng PaLM 2.
        
        Args:
            prompt: Chuỗi prompt đầu vào
            temperature: Nhiệt độ để kiểm soát tính ngẫu nhiên
            n: Số lượng phản hồi cần tạo
            max_trials: Số lần thử lại tối đa khi gặp lỗi
            failure_sleep_time: Thời gian chờ (giây) giữa các lần thử lại
            
        Returns:
            Danh sách các phản hồi được tạo ra
        """
        for _ in range(max_trials):
            try:
                results = palm.chat(
                    model=self.model_path,
                    prompt=prompt,
                    temperature=temperature,
                    candidate_count=n,
                )
                return [results.candidates[i]['content'] for i in range(n)]
            except Exception as e:
                logging.warning(
                    f"PaLM2 API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" " for _ in range(n)]
    
    def generate_batch(self, prompts, temperature=0, n=1, max_trials=1, failure_sleep_time=1):
        """
        Tạo phản hồi cho một loạt các prompt đầu vào sử dụng PaLM 2.
        
        Args:
            prompts: Danh sách các prompt đầu vào
            temperature: Nhiệt độ để kiểm soát tính ngẫu nhiên
            n: Số lượng phản hồi cần tạo cho mỗi prompt
            max_trials: Số lần thử lại tối đa khi gặp lỗi
            failure_sleep_time: Thời gian chờ (giây) giữa các lần thử lại
            
        Returns:
            Danh sách các phản hồi tương ứng với các prompt
        """
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, temperature, n,
                                       max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results

class ClaudeLLM(LLM):
    """
    Lớp đại diện cho mô hình Claude của Anthropic.
    Sử dụng API Anthropic để tạo phản hồi.
    """
    def __init__(self,
                 model_path='claude-instant-1.2',
                 api_key=None
                ):
        """
        Khởi tạo mô hình Claude.
        
        Args:
            model_path: Tên mô hình Claude (mặc định là 'claude-instant-1.2')
            api_key: Khóa API để truy cập Claude
        """
        super().__init__()
        
        if len(api_key) != 108:
            raise ValueError('invalid Claude API key')
        
        self.model_path = model_path
        self.api_key = api_key
        self.anthropic = Anthropic(
            api_key=self.api_key
        )

    def generate(self, prompt, max_tokens=512, max_trials=1, failure_sleep_time=1):
        """
        Tạo phản hồi từ prompt đầu vào sử dụng Claude.
        
        Args:
            prompt: Chuỗi prompt đầu vào
            max_tokens: Số lượng token tối đa trong phản hồi
            max_trials: Số lần thử lại tối đa khi gặp lỗi
            failure_sleep_time: Thời gian chờ (giây) giữa các lần thử lại
            
        Returns:
            Danh sách chứa phản hồi được tạo ra
        """
        
        for _ in range(max_trials):
            try:
                completion = self.anthropic.completions.create(
                    model=self.model_path,
                    max_tokens_to_sample=300,
                    prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
                )
                return [completion.completion]
            except Exception as e:
                logging.warning(
                    f"Claude API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" "]
    
    def generate_batch(self, prompts, max_tokens=512, max_trials=1, failure_sleep_time=1):
        """
        Tạo phản hồi cho một loạt các prompt đầu vào sử dụng Claude.
        
        Args:
            prompts: Danh sách các prompt đầu vào
            max_tokens: Số lượng token tối đa trong mỗi phản hồi
            max_trials: Số lần thử lại tối đa khi gặp lỗi
            failure_sleep_time: Thời gian chờ (giây) giữa các lần thử lại
            
        Returns:
            Danh sách các phản hồi tương ứng với các prompt
        """
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, max_tokens,
                                       max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results

class OpenAILLM(LLM):
    """
    Lớp đại diện cho mô hình OpenAI (GPT-3.5, GPT-4).
    Sử dụng API OpenAI để tạo phản hồi.
    """
    def __init__(self,
                 model_path,
                 api_key=None,
                 system_message=None
                ):
        """
        Khởi tạo mô hình OpenAI.
        
        Args:
            model_path: Tên mô hình OpenAI ('gpt-3.5-turbo' hoặc 'gpt-4')
            api_key: Khóa API để truy cập OpenAI
            system_message: Thông điệp hệ thống tùy chỉnh
        """
        super().__init__()

        if not api_key.startswith('sk-'):
            raise ValueError('OpenAI API key should start with sk-')
        if model_path not in ['gpt-3.5-turbo', 'gpt-4']:
            raise ValueError(
                'OpenAI model path should be gpt-3.5-turbo or gpt-4')
        self.client = OpenAI(api_key = api_key)
        self.model_path = model_path
        self.system_message = system_message if system_message is not None else "You are a helpful assistant."

    def generate(self, prompt, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        """
        Tạo phản hồi từ prompt đầu vào sử dụng OpenAI.
        
        Args:
            prompt: Chuỗi prompt đầu vào
            temperature: Nhiệt độ để kiểm soát tính ngẫu nhiên
            max_tokens: Số lượng token tối đa trong phản hồi
            n: Số lượng phản hồi cần tạo
            max_trials: Số lần thử lại tối đa khi gặp lỗi
            failure_sleep_time: Thời gian chờ (giây) giữa các lần thử lại
            
        Returns:
            Danh sách các phản hồi được tạo ra
        """
        for _ in range(max_trials):
            try:
                results = self.client.chat.completions.create(
                    model=self.model_path,
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                )
                return [results.choices[i].message.content for i in range(n)]
            except Exception as e:
                logging.warning(
                    f"OpenAI API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" " for _ in range(n)]

    def generate_batch(self, prompts, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        """
        Tạo phản hồi cho một loạt các prompt đầu vào sử dụng OpenAI.
        
        Args:
            prompts: Danh sách các prompt đầu vào
            temperature: Nhiệt độ để kiểm soát tính ngẫu nhiên
            max_tokens: Số lượng token tối đa trong mỗi phản hồi
            n: Số lượng phản hồi cần tạo cho mỗi prompt
            max_trials: Số lần thử lại tối đa khi gặp lỗi
            failure_sleep_time: Thời gian chờ (giây) giữa các lần thử lại
            
        Returns:
            Danh sách các phản hồi tương ứng với các prompt
        """
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, temperature, max_tokens, n,
                                       max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results
