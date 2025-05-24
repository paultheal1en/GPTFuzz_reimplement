from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch


class Predictor:
    """
    Lớp cơ sở cho các bộ dự đoán.
    Định nghĩa các phương thức chung mà tất cả các bộ dự đoán cần triển khai.
    """
    def __init__(self, path):
        """
        Khởi tạo đối tượng Predictor.
        
        Args:
            path: Đường dẫn đến mô hình
        """
        self.path = path

    def predict(self, sequences):
        """
        Phương thức trừu tượng để dự đoán kết quả từ chuỗi đầu vào.
        Cần được triển khai bởi các lớp con.
        
        Args:
            sequences: Danh sách các chuỗi đầu vào
            
        Raises:
            NotImplementedError: Khi phương thức chưa được triển khai
        """
        raise NotImplementedError("Predictor must implement predict method.")


class RoBERTaPredictor(Predictor):
    """
    Bộ dự đoán sử dụng mô hình RoBERTa.
    Kế thừa từ lớp Predictor và triển khai phương thức predict.
    """
    def __init__(self, path, device='cuda'):
        """
        Khởi tạo bộ dự đoán RoBERTa.
        
        Args:
            path: Đường dẫn đến mô hình RoBERTa
            device: Thiết bị để chạy mô hình ('cuda' hoặc 'cpu')
        """
        super().__init__(path)
        self.device = device
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.path).to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.path)

    def predict(self, sequences):
        """
        Dự đoán kết quả từ chuỗi đầu vào sử dụng mô hình RoBERTa.
        
        Args:
            sequences: Danh sách các chuỗi đầu vào
            
        Returns:
            Danh sách các lớp được dự đoán
        """
        inputs = self.tokenizer(sequences, padding=True, truncation=True,
                                max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        _, predicted_classes = torch.max(predictions, dim=1)
        predicted_classes = predicted_classes.cpu().tolist()
        return predicted_classes
