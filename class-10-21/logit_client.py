"""
Client library for interacting with the logit server.

This provides a simple Python interface for sending token IDs to the server
and receiving logits back, with automatic serialization/deserialization using
safetensors format.
"""
import torch
import requests
import safetensors.torch
from typing import Optional


class LogitClient:
    """Client for the logit server."""
    
    def __init__(self, base_url: str, timeout: Optional[float] = 30.0):
        """
        Initialize the logit client.
        
        Args:
            base_url: Server base URL (e.g., "http://localhost:8000")
            timeout: Request timeout in seconds (default: 30.0)
        """
        self.base_url = base_url
        self.timeout = timeout
    
    def get_logits(self, model_name: str, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get logits for the next token given a sequence of input token IDs.
        
        Args:
            model_name: Name of the model to use for inference
            input_ids: 1D LongTensor of token IDs on CPU
        
        Returns:
            1D tensor of bfloat16 logits for the next token (on CPU)
        
        Raises:
            ValueError: If input_ids is not a valid tensor
            requests.HTTPError: If the server returns an error
            requests.RequestException: If there's a network error
        """
        # Validate input
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("input_ids must be a torch.Tensor")
        
        if input_ids.dim() != 1:
            raise ValueError(f"input_ids must be 1D, got {input_ids.dim()}D")
        
        if input_ids.dtype != torch.long:
            raise ValueError(f"input_ids must be LongTensor (int64), got {input_ids.dtype}")
        
        # Ensure tensor is on CPU for serialization
        input_ids_cpu = input_ids.cpu()
        
        # Serialize input to safetensors
        input_data = {"input_ids": input_ids_cpu}
        binary_data = safetensors.torch.save(input_data)
        
        # Make request
        url = f"{self.base_url}/logits/{model_name}"
        response = requests.post(
            url,
            data=binary_data,
            headers={"Content-Type": "application/octet-stream"},
            timeout=self.timeout
        )
        
        # Check for errors
        response.raise_for_status()
        
        # Deserialize output
        output_tensors = safetensors.torch.load(response.content)
        logits = output_tensors["logits"]
        
        return logits
    
    def get_all_logits(self, model_name: str, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get logits at all positions in the sequence.
        
        Args:
            model_name: Name of the model to use for inference
            input_ids: 1D LongTensor of token IDs on CPU
        
        Returns:
            2D tensor of bfloat16 logits, shape [seq_len, vocab_size] (on CPU)
        
        Raises:
            ValueError: If input_ids is not a valid tensor
            requests.HTTPError: If the server returns an error
            requests.RequestException: If there's a network error
        """
        # Validate input
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("input_ids must be a torch.Tensor")
        
        if input_ids.dim() != 1:
            raise ValueError(f"input_ids must be 1D, got {input_ids.dim()}D")
        
        if input_ids.dtype != torch.long:
            raise ValueError(f"input_ids must be LongTensor (int64), got {input_ids.dtype}")
        
        # Ensure tensor is on CPU for serialization
        input_ids_cpu = input_ids.cpu()
        
        # Serialize input to safetensors
        input_data = {"input_ids": input_ids_cpu}
        binary_data = safetensors.torch.save(input_data)
        
        # Make request
        url = f"{self.base_url}/all_logits/{model_name}"
        response = requests.post(
            url,
            data=binary_data,
            headers={"Content-Type": "application/octet-stream"},
            timeout=self.timeout
        )
        
        # Check for errors
        response.raise_for_status()
        
        # Deserialize output
        output_tensors = safetensors.torch.load(response.content)
        logits = output_tensors["logits"]
        
        return logits

