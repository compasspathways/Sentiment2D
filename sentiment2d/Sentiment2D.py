import torch
from transformers import pipeline


class Sentiment2D:
    """Sentiment2D is a callable class that takes a string and returns a valence, arousal pair.
    It is intended to take one sentence/utterance at a time. Valence represents the positivity
    or negativity of the input in the usual sense of 'sentiment'. Arousal measures the level of
    excitedness/calmness.

    Note:
        Sentiment2D uses the zero-shot classifier available on hugginface.co as
        facebook/bart-large-mnli.

    Attributes:
        classes: This is the list of the classes used for the zero-shot classifier.
        model_dir: This is an optional argument to __init__ that allows the use of local copy of
                         facebook/bart-large-mnli.
    """

    def __init__(self, model_dir=None, device=None):
        """Instantiate class instance

        :param model_dir: local directory for the model cache. Set to None to not cache the model.
        :param device: Device to run the model. The default of None will auto-detect a GPU and use
                       it, falling back to CPU if no GPU is found. To force CPU use, set to -1.
        """
        # the classes represent x (valence) and y (arousal) scores of (1,1), (-1,1), (-1,-1), (1,-1)
        self.classes = ["happy", "angry", "gloomy", "calm"]
        self.model_dir = model_dir
        device = device if device else torch.cuda.current_device() if torch.cuda.is_available() else -1
        if model_dir:
            try:
                self.classifier = pipeline(
                    "zero-shot-classification", model=model_dir, tokenizer=model_dir, device=device
                )
            except FileNotFoundError:
                self.classifier = pipeline(
                    "zero-shot-classification", model="facebook/bart-large-mnli", device=device
                )
                self.classifier.save_pretrained(model_dir)
        else:
            self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

    def get_utterance_class_scores(self, utterance):
        cl_out = self.classifier(utterance, self.classes)
        if isinstance(cl_out, list):
            scores = [dict(zip(c["labels"], c["scores"])) for c in cl_out]
        else:
            scores = dict(zip(cl_out["labels"], cl_out["scores"]))
        return scores

    def get_utterance_valence_arousal(self, utterance):
        scores = self.get_utterance_class_scores(utterance)
        if isinstance(scores, list):
            valence = [
                s[self.classes[0]] + s[self.classes[3]] - s[self.classes[1]] - s[self.classes[2]] for s in scores
            ]
            arousal = [
                s[self.classes[0]] + s[self.classes[1]] - s[self.classes[3]] - s[self.classes[2]] for s in scores
            ]
        else:
            valence = (
                scores[self.classes[0]]
                + scores[self.classes[3]]
                - scores[self.classes[1]]
                - scores[self.classes[2]]
            )
            arousal = (
                scores[self.classes[0]]
                + scores[self.classes[1]]
                - scores[self.classes[3]]
                - scores[self.classes[2]]
            )
        return (valence, arousal)

    def __call__(self, utterance):
        return self.get_utterance_valence_arousal(utterance)
