"""Scorer"""
import nltk
from evaluator.predictor import Predictor
from utils.constants import ENTITY_TOKEN

class BleuScorer(object):
    """Blue scorer class"""
    def __init__(self):
        self.results = []
        self.score = 0
        self.m_score = 0
        self.instances = 0

    def example_score(self, reference, hypothesis):
        """Calculate blue score for one example"""
        return nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)

    def meteor_score(self, reference, hypothesis):
        reference_string = ' '.join(reference)
        hypothesis_string = ' '.join(hypothesis)
        return nltk.translate.meteor_score.single_meteor_score(reference=reference_string, hypothesis=hypothesis_string)

    def data_score(self, data_ques, data_quer, predictor):
        """Score complete list of data"""
        for _,batch in enumerate(zip(data_ques, data_quer)):
            example_ques, example_quer = batch
            reference = [t.lower() for t in example_ques.trg]
            reference_query = [[t.lower() for t in example_quer.trg]]
            hypothesis = predictor.predict(example_ques.src, example_quer.src)
            blue_score = self.example_score(reference, hypothesis)
            m_score_t = self.meteor_score(reference, hypothesis)
            self.results.append({
                'reference': reference,
                'hypothesis': hypothesis,
                'blue_score': blue_score,
                'reference_query': reference_query
            })
            self.score += blue_score
            self.m_score += m_score_t
            self.instances += 1

        return self.results, self.score / self.instances

    def average_score(self):
        """Return bleu average score"""
        return self.score / self.instances

    def average_meteor_score(self):
        return self.m_score / self.instances

    def reset(self):
        """Reset object properties"""
        self.results = []
        self.score = 0
        self.instances = 0
