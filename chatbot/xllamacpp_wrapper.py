"""
XLlamaCPP Wrapper - Provides llama-cpp-python compatible interface using xllamacpp.
Uses direct handle_chat_completions calls (non-streaming internally, simulates streaming externally).
"""

import sys
from typing import List, Dict, Generator, Any, Union, Optional

try:
    import xllamacpp as xlc
    XLLAMACPP_AVAILABLE = True
except ImportError:
    xlc = None
    XLLAMACPP_AVAILABLE = False
    print("WARNING: xllamacpp not installed. Install with: pip install xllamacpp")

# Global server instance to avoid re-initialization issues
_server_instance = None
_current_model_path = None


class XLlamaCPPWrapper:
    """
    Wrapper that provides a llama-cpp-python compatible interface using xllamacpp.
    """
    
    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1, verbose: bool = False):
        global _server_instance, _current_model_path
        
        if not XLLAMACPP_AVAILABLE:
            raise ImportError("xllamacpp is not installed. Install with: pip install xllamacpp")
        
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        
        # Reuse existing server if same model
        if _server_instance is not None and _current_model_path == model_path:
            print(f"[XLlamaCPP] Reusing existing server for: {model_path}")
            self.server = _server_instance
            return
        
        # Initialize xllamacpp Server with CommonParams
        params = xlc.CommonParams()
        params.model.path = model_path
        params.n_ctx = n_ctx
        
        # GPU layers: -1 means all layers on GPU
        if n_gpu_layers == -1:
            params.n_gpu_layers = 999
        else:
            params.n_gpu_layers = n_gpu_layers
        
        # Create the server
        print(f"[XLlamaCPP] Loading model: {model_path}")
        self.server = xlc.Server(params)
        
        # Cache globally
        _server_instance = self.server
        _current_model_path = model_path
        print(f"[XLlamaCPP] Model loaded successfully")

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stream: bool = False,
        repeat_penalty: float = 1.1,
        **kwargs
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Mimics llama_cpp.Llama.create_chat_completion interface.
        """
        
        # Build request payload (OpenAI-compatible format) - ALWAYS non-streaming internally
        request = {
            "model": "local-model",
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,  # xllamacpp handles this differently
        }
        
        if max_tokens:
            request["max_tokens"] = max_tokens
        
        if repeat_penalty and repeat_penalty != 1.0:
            request["frequency_penalty"] = max(0.0, repeat_penalty - 1.0)
        
        try:
            # Call xllamacpp directly
            response = self.server.handle_chat_completions(request)
            
            if stream:
                # Simulate streaming by yielding the full response as one chunk
                return self._simulate_stream(response)
            else:
                return response
                
        except Exception as e:
            print(f"[XLlamaCPP] Error: {e}", file=sys.stderr)
            raise RuntimeError(f"XLlamaCPP completion failed: {e}")
    
    def _simulate_stream(self, response: Dict) -> Generator[Dict[str, Any], None, None]:
        """Simulate streaming by yielding the full response content character by character."""
        try:
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            if content:
                # Yield in small chunks for responsive UI
                chunk_size = 4
                for i in range(0, len(content), chunk_size):
                    chunk_text = content[i:i+chunk_size]
                    yield {
                        'choices': [{
                            'delta': {'content': chunk_text}
                        }]
                    }
        except Exception as e:
            print(f"[XLlamaCPP] Stream simulation error: {e}", file=sys.stderr)
            raise


def is_xllamacpp_available() -> bool:
    """Check if xllamacpp is available."""
    return XLLAMACPP_AVAILABLE
