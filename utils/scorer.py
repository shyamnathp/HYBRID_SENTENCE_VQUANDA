"""Scorer"""
import nltk

class BleuScorer(object):
    """Blue scorer class"""
    def __init__(self):
        self.results = []
        self.score = 0
        self.instances = 0

    def example_score(self, reference, hypothesis):
        """Calculate blue score for one example"""
        return nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)

    def data_score(self, data_ques, data_quer, predictor):
        """Score complete list of data"""
        for _,batch in enumerate(zip(data_ques, data_quer)):
            example_ques, example_quer = batch
            reference = [t.lower() for t in example_ques.trg]
            reference_query = [[t.lower() for t in example_quer.trg]]
            hypothesis = predictor.predict(example_ques.src, example_quer.src)
            blue_score = self.example_score(reference, hypothesis)
            self.results.append({
                'reference': reference,
                'hypothesis': hypothesis,
                'blue_score': blue_score,
                'reference_query': reference_query
            })
            self.score += blue_score
            self.instances += 1

        return self.results, self.score / self.instances

    def average_score(self):
        """Return bleu average score"""
        return self.score / self.instances

    def reset(self):
        """Reset object properties"""
        self.results = []
        self.score = 0
        self.instances = 0
