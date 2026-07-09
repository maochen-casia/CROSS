import logging
import os

class Logger:
    def __init__(self, log_dir, log_level=logging.INFO):

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f"log.txt")

        self.logger = logging.getLogger('Trainer')
        self.logger.setLevel(log_level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log_metrics(self, epoch, metrics_dict):

        log_str = f"Epoch {epoch:03d}"
        for key, value in metrics_dict.items():
            log_str += f" | {key}: {value:.4f}"
        self.info(log_str)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)