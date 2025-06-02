import os
import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import pandas as pd
import sys
import logging

# Thêm thư mục gốc vào sys.path để import các module từ gptfuzzer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gptfuzzer.llm import OpenAILLM
from gptfuzzer.utils.predict import RoBERTaPredictor
from gptfuzzer.fuzzer.selection import MCTSExploreSelectPolicy
from gptfuzzer.fuzzer.mutator import (
    MutateRandomSinglePolicy, OpenAIMutatorCrossOver, OpenAIMutatorExpand,
    OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten
)
from gptfuzzer.fuzzer import GPTFuzzer

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = ""
        
    def write(self, string):
        self.buffer += string
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)

    def flush(self):
        pass

class JailbreakDemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenAI Jailbreak Demo")
        self.root.geometry("800x600")
        
        # Biến cờ để kiểm tra xem models đã được khởi tạo chưa
        self.models_initialized = False
        
        # Tạo widgets
        self.create_widgets()
        
        # Vô hiệu hóa nút jailbreak cho đến khi models được khởi tạo
        self.jailbreak_button.config(state=tk.DISABLED)
        
        # Khởi tạo models
        self.initialize_models_thread = threading.Thread(target=self.initialize_models)
        self.initialize_models_thread.daemon = True
        self.initialize_models_thread.start()
        
        # Hiển thị thông báo đang khởi tạo
        self.result_text.insert(tk.END, "Đang khởi tạo models, vui lòng đợi...\n")
    
    def initialize_models(self):
        try:
            # Khởi tạo OpenAI model
            # Lưu ý: Bạn cần thay 'openaikey' bằng API key thực của bạn
            openai_model_path = 'gpt-3.5-turbo'
            self.openai_model = OpenAILLM(openai_model_path, os.environ.get('OPENAI_API_KEY', 'openaikey'))
            
            # Khởi tạo RoBERTa model
            self.roberta_model = RoBERTaPredictor('hubert233/GPTFuzz', device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu')
            
            # Tải initial seed
            seed_path = 'datasets/prompts/GPTFuzzer.csv'
            self.initial_seed = pd.read_csv(seed_path)['text'].tolist()
            
            # Khởi tạo mutators
            self.mutators = [
                OpenAIMutatorCrossOver(self.openai_model, temperature=0.0),
                OpenAIMutatorExpand(self.openai_model, temperature=1.0),
                OpenAIMutatorGenerateSimilar(self.openai_model, temperature=0.5),
                OpenAIMutatorRephrase(self.openai_model),
                OpenAIMutatorShorten(self.openai_model)
            ]
            
            # Đánh dấu là đã khởi tạo xong
            self.models_initialized = True
            
            print("Khởi tạo models hoàn tất!")
            self.root.after(0, lambda: self.result_text.insert(tk.END, "\nKhởi tạo models hoàn tất! Bạn có thể bắt đầu jailbreak.\n"))
            
            # Kích hoạt lại nút jailbreak
            self.root.after(0, lambda: self.jailbreak_button.config(state=tk.NORMAL))
        except Exception as e:
            print(f"Lỗi khi khởi tạo models: {str(e)}")
            self.root.after(0, lambda: self.result_text.insert(tk.END, f"\nLỗi khi khởi tạo models: {str(e)}\n"))
    
    def create_widgets(self):
        # Frame chính
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Label và Entry cho prompt
        prompt_frame = ttk.Frame(main_frame)
        prompt_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(prompt_frame, text="Nhập prompt:").pack(side=tk.LEFT, padx=5)
        
        self.prompt_entry = ttk.Entry(prompt_frame, width=50)
        self.prompt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.jailbreak_button = ttk.Button(prompt_frame, text="Jailbreak!", command=self.start_jailbreak)
        self.jailbreak_button.pack(side=tk.LEFT, padx=5)
        
        # Khu vực hiển thị kết quả
        result_frame = ttk.LabelFrame(main_frame, text="Kết quả", padding=5)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.result_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, height=10)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # Khu vực hiển thị log
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Chuyển hướng stdout sang log_text
        self.stdout_redirect = RedirectText(self.log_text)
        sys.stdout = self.stdout_redirect
    
    def start_jailbreak(self):
        prompt = self.prompt_entry.get()
        if not prompt:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Vui lòng nhập prompt!")
            return
        
        # Kiểm tra xem models đã được khởi tạo chưa
        if not self.models_initialized:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Vui lòng đợi cho đến khi models được khởi tạo hoàn tất!")
            return
        
        # Vô hiệu hóa nút jailbreak trong quá trình xử lý
        self.jailbreak_button.config(state=tk.DISABLED)
        
        # Xóa kết quả cũ
        self.result_text.delete(1.0, tk.END)
        
        # Tạo và chạy thread để không block giao diện
        thread = threading.Thread(target=self.run_jailbreak, args=(prompt,))
        thread.daemon = True
        thread.start()
        
    def run_jailbreak(self, prompt):
        try:
            print(f"Bắt đầu jailbreak với prompt: {prompt}")
            
            # Tạo thư mục kết quả nếu chưa tồn tại
            os.makedirs("results", exist_ok=True)
            
            # Khởi tạo GPTFuzzer
            fuzzer = GPTFuzzer(
                questions=[prompt],  # chỉ 1 câu hỏi
                target=self.openai_model,
                predictor=self.roberta_model,
                initial_seed=self.initial_seed,
                mutate_policy=MutateRandomSinglePolicy(self.mutators, concatentate=True),
                select_policy=MCTSExploreSelectPolicy(),
                energy=1,
                max_jailbreak=1,
                max_query=500,  
                generate_in_batch=False,
                result_file="results/user_prompt_result.csv"
            )
            
            print(f"Đang chạy fuzzing...")
            fuzzer.run()
            print(f"Fuzzing hoàn tất!")
            
            # Đọc kết quả từ file
            try:
                jailbreak_results = pd.read_csv("results/user_prompt_result.csv")
                if not jailbreak_results.empty:
                    # Lấy kết quả jailbreak đầu tiên
                    jailbreak_result = jailbreak_results.iloc[0]
                    self.root.after(0, lambda: self.update_result(
                        f"Jailbreak thành công!\n\nPrompt jailbreak:\n{jailbreak_result['prompt']}\n\nResponse từ OpenAI:\n\n{jailbreak_result['response']}"))
                else:
                    self.root.after(0, lambda: self.update_result(
                        "Không tìm thấy jailbreak nào. Hãy thử lại với prompt khác."))
            except Exception as e:
                print(f"Lỗi khi đọc kết quả: {str(e)}")
                self.root.after(0, lambda: self.update_result(f"Lỗi khi đọc kết quả: {str(e)}"))
        except Exception as e:
            print(f"Lỗi khi thực hiện jailbreak: {str(e)}")
            self.root.after(0, lambda: self.update_result(f"Lỗi khi thực hiện jailbreak: {str(e)}"))
        finally:
            # Kích hoạt lại nút jailbreak
            self.root.after(0, lambda: self.jailbreak_button.config(state=tk.NORMAL))
    
    def update_result(self, text):
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)

if __name__ == "__main__":
    root = tk.Tk()
    app = JailbreakDemoApp(root)
    root.mainloop()