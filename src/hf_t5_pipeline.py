"""
T5 (seq2seq) pipeline compatible with LangChain HuggingFacePipeline.
Uses AutoModelForSeq2SeqLM directly because transformers 5.x removed
the "text2text-generation" pipeline task.
"""


def get_t5_pipeline(
    model_id: str = "google-t5/t5-small",
    max_new_tokens: int = 150,
    device_map: str | None = None,
):
    """
    Return a pipeline-like object for T5 that HuggingFacePipeline can use.
    Has .task = "text2text-generation" and __call__(prompts, **kwargs) returning
    a list of {"generated_text": "..."}.
    """
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch
    except ImportError:
        raise ImportError("Install: pip install transformers torch")

    device = "cuda" if (torch.cuda.is_available() and device_map == "auto") else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model = model.to(device)
    model.eval()

    class T5PipelineLike:
        task = "text2text-generation"

        def __init__(self, tok, mdl, max_new_tokens: int):
            self.tokenizer = tok
            self.model = mdl
            self.max_new_tokens = max_new_tokens

        def __call__(self, prompts, max_new_tokens=None, **kwargs):
            if isinstance(prompts, str):
                prompts = [prompts]
            max_tok = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
            out = []
            for prompt in prompts:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(self.model.device)
                with torch.no_grad():
                    generated = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tok,
                        **kwargs,
                    )
                text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                out.append({"generated_text": text})
            return out

    return T5PipelineLike(tokenizer, model, max_new_tokens=max_new_tokens)
