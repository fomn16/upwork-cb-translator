import threading
from typing import List, Optional
import stanza


class SentenceSplitter:
    """
    Sentence tokenizer that wraps Stanza's tokenizer.
    Exposes a `split(text) -> List[str]` method compatible with Moses-style usage.
    """

    SUPPORTED_LANGS = [
        # Major Indian languages with UD/Stanza models
        "hi",  # Hindi
        "bn",  # Bengali
        "ta",  # Tamil
        "te",  # Telugu
        "ml",  # Malayalam
        "kn",  # Kannada
        "mr",  # Marathi
        "gu",  # Gujarati
        "or",  # Odia
        "pa",  # Punjabi
        "ur",  # Urdu
        "ne",  # Nepali
        "sa",  # Sanskrit
        "as",  # Assamese
        # Other
        "en",  # English
        "pt",  # Portuguese
    ]

    def __init__(
        self,
        lang: str,
        use_gpu: bool = False,
        processors: str = "tokenize",  # only tokenizer needed
        tokenize_no_ssplit: bool = False,  # ensure sentence splitting is ON
        verbose: bool = False,
    ):
        """
        lang: ISO code, e.g., "hi" (Hindi), "bn", "ta", "te", "ml", "kn", "mr", "gu", "or", "pa", etc.
        use_gpu: set True if CUDA is available and you prefer GPU (not necessary for tokenize-only).
        tokenize_no_ssplit: must be False to enable sentence splitting.
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.processors = processors
        self.tokenize_no_ssplit = tokenize_no_ssplit
        self.verbose = verbose

        self._pipeline: Optional[stanza.Pipeline] = None
        self._lock = threading.Lock()

        # Ensure all required models are downloaded once
        self._download_required_models()

        # Initialize pipeline immediately
        self._ensure_pipeline()

    def _download_required_models(self):
        """
        Download English, Portuguese, and all supported Indian language models.
        Safe to call multiple times; stanza.download() skips if already present.
        """
        for lang in self.SUPPORTED_LANGS:
            try:
                stanza.download(lang, processors="tokenize", verbose=self.verbose)
            except Exception as e:
                if self.verbose:
                    print(f"[SentenceSplitter] Warning: could not download {lang}: {e}")

    def _ensure_pipeline(self):
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    if self.verbose:
                        print(f"[SentenceSplitter] Initializing pipeline for {self.lang}")
                    self._pipeline = stanza.Pipeline(
                        lang=self.lang,
                        processors=self.processors,
                        tokenize_no_ssplit=self.tokenize_no_ssplit,
                        use_gpu=self.use_gpu,
                        verbose=self.verbose,
                    )

    def split(self, text: str) -> List[str]:
        """
        Return a list of sentence strings.
        """
        if not text:
            return []
        self._ensure_pipeline()
        doc = self._pipeline(text)
        sents = []
        for sent in doc.sentences:
            if getattr(sent, "text", None):
                sents.append(sent.text.strip())
            else:
                toks = [w.text for w in sent.tokens]
                sents.append(" ".join(toks).strip())
        return [s for s in sents if s]  # remove empty strings