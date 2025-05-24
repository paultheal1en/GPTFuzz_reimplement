import random
import numpy as np

from gptfuzzer.fuzzer import GPTFuzzer, PromptNode


class SelectPolicy:
    """
    Lớp cơ sở cho tất cả các chính sách lựa chọn.
    Các chính sách lựa chọn được sử dụng để quyết định prompt nào sẽ được đột biến tiếp theo.
    """
    def __init__(self, fuzzer: GPTFuzzer):
        """
        Khởi tạo chính sách lựa chọn.
        
        Args:
            fuzzer: Đối tượng GPTFuzzer quản lý chính sách này
        """
        self.fuzzer = fuzzer

    def select(self) -> PromptNode:
        """
        Phương thức trừu tượng để lựa chọn một nút prompt.
        Cần được triển khai bởi các lớp con.
        
        Returns:
            Nút prompt được chọn để đột biến tiếp theo
        """
        raise NotImplementedError(
            "SelectPolicy must implement select method.")

    def update(self, prompt_nodes: 'list[PromptNode]'):
        """
        Cập nhật trạng thái của chính sách lựa chọn sau khi đánh giá các prompt.
        
        Args:
            prompt_nodes: Danh sách các nút prompt đã được đánh giá
        """
        pass


class RoundRobinSelectPolicy(SelectPolicy):
    """
    Chính sách lựa chọn theo vòng tròn.
    Lựa chọn các prompt theo thứ tự tuần tự, quay vòng.
    """
    def __init__(self, fuzzer: GPTFuzzer = None):
        """
        Khởi tạo chính sách lựa chọn vòng tròn.
        
        Args:
            fuzzer: Đối tượng GPTFuzzer quản lý chính sách này
        """
        super().__init__(fuzzer)
        self.index: int = 0  # Chỉ số của prompt hiện tại

    def select(self) -> PromptNode:
        """
        Lựa chọn prompt tiếp theo theo thứ tự vòng tròn.
        
        Returns:
            Nút prompt được chọn
        """
        seed = self.fuzzer.prompt_nodes[self.index]
        seed.visited_num += 1  # Tăng số lần truy cập
        return seed

    def update(self, prompt_nodes: 'list[PromptNode]'):
        """
        Cập nhật chỉ số để chọn prompt tiếp theo trong danh sách.
        
        Args:
            prompt_nodes: Danh sách các nút prompt đã được đánh giá
        """
        self.index = (self.index - 1 + len(self.fuzzer.prompt_nodes)
                      ) % len(self.fuzzer.prompt_nodes)


class RandomSelectPolicy(SelectPolicy):
    """
    Chính sách lựa chọn ngẫu nhiên.
    Lựa chọn các prompt một cách ngẫu nhiên từ danh sách.
    """
    def __init__(self, fuzzer: GPTFuzzer = None):
        """
        Khởi tạo chính sách lựa chọn ngẫu nhiên.
        
        Args:
            fuzzer: Đối tượng GPTFuzzer quản lý chính sách này
        """
        super().__init__(fuzzer)

    def select(self) -> PromptNode:
        """
        Lựa chọn một prompt ngẫu nhiên từ danh sách.
        
        Returns:
            Nút prompt được chọn ngẫu nhiên
        """
        seed = random.choice(self.fuzzer.prompt_nodes)
        seed.visited_num += 1  # Tăng số lần truy cập
        return seed


class UCBSelectPolicy(SelectPolicy):
    """
    Chính sách lựa chọn dựa trên thuật toán Upper Confidence Bound (UCB).
    Cân bằng giữa khai thác (exploitation) và khám phá (exploration).
    """
    def __init__(self,
                 explore_coeff: float = 1.0,
                 fuzzer: GPTFuzzer = None):
        """
        Khởi tạo chính sách lựa chọn UCB.
        
        Args:
            explore_coeff: Hệ số khám phá, điều chỉnh mức độ ưu tiên khám phá
            fuzzer: Đối tượng GPTFuzzer quản lý chính sách này
        """
        super().__init__(fuzzer)

        self.step = 0  # Số bước đã thực hiện
        self.last_choice_index = None  # Chỉ số của lựa chọn gần nhất
        self.explore_coeff = explore_coeff  # Hệ số khám phá
        self.rewards = [0 for _ in range(len(self.fuzzer.prompt_nodes))]  # Phần thưởng cho mỗi prompt

    def select(self) -> PromptNode:
        """
        Lựa chọn prompt dựa trên điểm UCB.
        
        Returns:
            Nút prompt có điểm UCB cao nhất
        """
        if len(self.fuzzer.prompt_nodes) > len(self.rewards):
            self.rewards.extend(
                [0 for _ in range(len(self.fuzzer.prompt_nodes) - len(self.rewards))])

        self.step += 1
        scores = np.zeros(len(self.fuzzer.prompt_nodes))
        for i, prompt_node in enumerate(self.fuzzer.prompt_nodes):
            smooth_visited_num = prompt_node.visited_num + 1
            # Công thức UCB: phần thưởng trung bình + hệ số khám phá * căn(2 * ln(tổng số bước) / số lần truy cập)
            scores[i] = self.rewards[i] / smooth_visited_num + \
                self.explore_coeff * \
                np.sqrt(2 * np.log(self.step) / smooth_visited_num)

        self.last_choice_index = np.argmax(scores)  # Chọn prompt có điểm cao nhất
        self.fuzzer.prompt_nodes[self.last_choice_index].visited_num += 1
        return self.fuzzer.prompt_nodes[self.last_choice_index]

    def update(self, prompt_nodes: 'list[PromptNode]'):
        """
        Cập nhật phần thưởng cho prompt đã chọn dựa trên kết quả đánh giá.
        
        Args:
            prompt_nodes: Danh sách các nút prompt đã được đánh giá
        """
        succ_num = sum([prompt_node.num_jailbreak
                        for prompt_node in prompt_nodes])
        self.rewards[self.last_choice_index] += \
            succ_num / len(self.fuzzer.questions)


class MCTSExploreSelectPolicy(SelectPolicy):
    """
    Chính sách lựa chọn dựa trên thuật toán Monte Carlo Tree Search (MCTS).
    Xây dựng và khám phá cây tìm kiếm để tìm ra các prompt hiệu quả.
    """
    def __init__(self, fuzzer: GPTFuzzer = None, ratio=0.5, alpha=0.1, beta=0.2):
        """
        Khởi tạo chính sách lựa chọn MCTS.
        
        Args:
            fuzzer: Đối tượng GPTFuzzer quản lý chính sách này
            ratio: Tỷ lệ cân bằng giữa khai thác và khám phá
            alpha: Hệ số phạt cho cấp độ (level)
            beta: Phần thưởng tối thiểu sau khi áp dụng phạt
        """
        super().__init__(fuzzer)

        self.step = 0  # Số bước đã thực hiện
        self.mctc_select_path: 'list[PromptNode]' = []  # Đường dẫn lựa chọn trong cây MCTS
        self.last_choice_index = None  # Chỉ số của lựa chọn gần nhất
        self.rewards = []  # Phần thưởng cho mỗi prompt
        self.ratio = ratio  # Tỷ lệ cân bằng giữa khai thác và khám phá
        self.alpha = alpha  # Hệ số phạt cho cấp độ
        self.beta = beta   # Phần thưởng tối thiểu sau khi áp dụng phạt

    def select(self) -> PromptNode:
        """
        Lựa chọn prompt dựa trên thuật toán MCTS.
        
        Returns:
            Nút prompt được chọn từ cây MCTS
        """
        self.step += 1
        if len(self.fuzzer.prompt_nodes) > len(self.rewards):
            self.rewards.extend(
                [0 for _ in range(len(self.fuzzer.prompt_nodes) - len(self.rewards))])

        self.mctc_select_path.clear()
        # Chọn nút gốc tốt nhất từ các prompt ban đầu
        cur = max(
            self.fuzzer.initial_prompts_nodes,
            key=lambda pn:
            self.rewards[pn.index] / (pn.visited_num + 1) +
            self.ratio * np.sqrt(2 * np.log(self.step) /
                                 (pn.visited_num + 0.01))
        )
        self.mctc_select_path.append(cur)

        # Đi xuống cây cho đến khi không còn nút con hoặc dừng ngẫu nhiên
        while len(cur.child) > 0:
            if np.random.rand() < self.alpha:
                break
            cur = max(
                cur.child,
                key=lambda pn:
                self.rewards[pn.index] / (pn.visited_num + 1) +
                self.ratio * np.sqrt(2 * np.log(self.step) /
                                     (pn.visited_num + 0.01))
            )
            self.mctc_select_path.append(cur)

        # Tăng số lần truy cập cho tất cả các nút trong đường dẫn
        for pn in self.mctc_select_path:
            pn.visited_num += 1

        self.last_choice_index = cur.index
        return cur

    def update(self, prompt_nodes: 'list[PromptNode]'):
        """
        Cập nhật phần thưởng cho tất cả các nút trong đường dẫn MCTS.
        
        Args:
            prompt_nodes: Danh sách các nút prompt đã được đánh giá
        """
        succ_num = sum([prompt_node.num_jailbreak
                        for prompt_node in prompt_nodes])

        last_choice_node = self.fuzzer.prompt_nodes[self.last_choice_index]
        # Cập nhật phần thưởng từ dưới lên trên (từ lá đến gốc)
        for prompt_node in reversed(self.mctc_select_path):
            reward = succ_num / (len(self.fuzzer.questions)
                                 * len(prompt_nodes))
            self.rewards[prompt_node.index] += reward * \
                max(self.beta, (1 - 0.1 * last_choice_node.level))


class EXP3SelectPolicy(SelectPolicy):
    """
    Chính sách lựa chọn dựa trên thuật toán EXP3 (Exponential-weight algorithm for Exploration and Exploitation).
    Phù hợp cho các môi trường đối kháng hoặc không cố định.
    """
    def __init__(self,
                 gamma: float = 0.05,
                 alpha: float = 25,
                 fuzzer: GPTFuzzer = None):
        """
        Khởi tạo chính sách lựa chọn EXP3.
        
        Args:
            gamma: Tham số điều chỉnh mức độ khám phá
            alpha: Tham số học tập
            fuzzer: Đối tượng GPTFuzzer quản lý chính sách này
        """
        super().__init__(fuzzer)

        self.energy = self.fuzzer.energy  # Năng lượng từ fuzzer
        self.gamma = gamma  # Tham số điều chỉnh mức độ khám phá
        self.alpha = alpha  # Tham số học tập
        self.last_choice_index = None  # Chỉ số của lựa chọn gần nhất
        self.weights = [1. for _ in range(len(self.fuzzer.prompt_nodes))]  # Trọng số cho mỗi prompt
        self.probs = [0. for _ in range(len(self.fuzzer.prompt_nodes))]  # Xác suất chọn mỗi prompt

    def select(self) -> PromptNode:
        """
        Lựa chọn prompt dựa trên thuật toán EXP3.
        
        Returns:
            Nút prompt được chọn theo xác suất từ EXP3
        """
        if len(self.fuzzer.prompt_nodes) > len(self.weights):
            self.weights.extend(
                [1. for _ in range(len(self.fuzzer.prompt_nodes) - len(self.weights))])

        np_weights = np.array(self.weights)
        # Tính xác suất: kết hợp giữa khai thác (dựa trên trọng số) và khám phá đồng đều
        probs = (1 - self.gamma) * np_weights / np_weights.sum() + \
            self.gamma / len(self.fuzzer.prompt_nodes)

        # Chọn prompt theo xác suất
        self.last_choice_index = np.random.choice(
            len(self.fuzzer.prompt_nodes), p=probs)

        self.fuzzer.prompt_nodes[self.last_choice_index].visited_num += 1
        self.probs[self.last_choice_index] = probs[self.last_choice_index]

        return self.fuzzer.prompt_nodes[self.last_choice_index]

    def update(self, prompt_nodes: 'list[PromptNode]'):
        """
        Cập nhật trọng số cho prompt đã chọn dựa trên kết quả đánh giá.
        
        Args:
            prompt_nodes: Danh sách các nút prompt đã được đánh giá
        """
        succ_num = sum([prompt_node.num_jailbreak
                        for prompt_node in prompt_nodes])

        # Tính phần thưởng (r) và cập nhật trọng số
        r = 1 - succ_num / len(prompt_nodes)
        x = -1 * r / self.probs[self.last_choice_index]
        self.weights[self.last_choice_index] *= np.exp(
            self.alpha * x / len(self.fuzzer.prompt_nodes))
