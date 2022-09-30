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

    def __init__(self, model_dir=None):
        self.classes = ["happy", "angry", "gloomy", "calm"]
        self.model_dir = model_dir
        if model_dir:
            try:
                self.classifier = pipeline("zero-shot-classification", model=model_dir, tokenizer=model_dir)
            except FileNotFoundError:
                self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
                self.classifier.save_pretrained(model_dir)
        else:
            self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def get_utterance_class_scores(self, utterance):
        cl_out = self.classifier(utterance, self.classes)
        scores = dict(zip(cl_out["labels"], cl_out["scores"]))
        return scores

    def get_utterance_valence_arousal(self, utterance):
        scores = self.get_utterance_class_scores(utterance)
        valence = scores["happy"] + scores["calm"] - scores["angry"] - scores["gloomy"]
        arousal = scores["happy"] + scores["angry"] - scores["calm"] - scores["gloomy"]
        return (valence, arousal)

    def __call__(self, utterance):
        return self.get_utterance_valence_arousal(utterance)
