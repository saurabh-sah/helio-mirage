from abc import ABC, abstractmethod
import torch

class BaseDecoder(ABC):
    """
    Base interface for training-free decoding strategies (Task B).
    Ensures that VCD, OPERA, and layer-contrast methods can be swapped easily.
    """
    @abstractmethod
    def decode(self, model, input_ids, image_tensor, **kwargs) -> torch.Tensor:
        """
        Runs the decoding strategy.
        Args:
            model: The VLM model
            input_ids: Tokenized prompt
            image_tensor: The clean image tensor
            kwargs: Any additional strategy-specific parameters
        Returns:
            Generated output tokens
        """
        pass
