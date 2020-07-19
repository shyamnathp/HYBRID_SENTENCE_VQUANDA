"""VerbalDataset"""
import os
import re
import json
import random
from pathlib import Path
from . import InputExample

ENTITY_TOKEN = '<ent>'

class VQUANDAReader(object):
    QUERY_DICT = {
        'x': 'var_x',
        'uri': 'var_uri',
        '{': 'brack_open',
        '}': 'brack_close',
        '.': 'sep_dot',
        'COUNT(uri)': 'count_uri'
    }
    ROOT_PATH = Path(os.path.dirname(__file__))

    def __init__(self, dataset_folder):
        self.data_path = str(dataset_folder)

    def _prepare_query(self, query, cover_entities=True):
        """
        trasnform query from this:
        SELECT DISTINCT COUNT(?uri) WHERE { ?x <http://dbpedia.org/ontology/commander> <http://dbpedia.org/resource/Andrew_Jackson> . ?uri <http://dbpedia.org/ontology/knownFor> ?x  . }
        To this:
        select distinct count_uri where brack_open var_x commander Andrew Jackson sep_dot var_uri known for var_x sep_dot brack_close
        """
        query = query.replace('\n', ' ')\
                     .replace('\t', '')\
                     .replace('?', '')\
                     .replace('{?', '{ ?')\
                     .replace('>}', '> }')\
                     .replace('uri}', 'uri }').strip()
        query = query.split()
        new_query = []
        for q in query:
            if q in self.QUERY_DICT:
                q = self.QUERY_DICT[q]
            if 'http' in q:
                if 'dbpedia.org/ontology' in q or 'dbpedia.org/property' in q:
                    q = q.rsplit('/', 1)[-1].lstrip('<').rstrip('>')
                    q = filter(None, re.split("([A-Z][^A-Z]*)", q))
                    q = ' '.join(q)
                    #q = ENTITY_TOKEN
                elif 'www.w3.org/1999/02/22-rdf-syntax-ns#type' in q:
                    q = 'type'
                elif cover_entities:
                    q = ENTITY_TOKEN
                else:
                    q = q.rsplit('/', 1)[-1].lstrip('<').rstrip('>').replace('_', ' ')
            new_query.append(q.lower())

        assert new_query[-1] == 'brack_close', 'Query not ending with a bracket.'
        return ' '.join(new_query)

    def get_examples(self, filename, max_examples=0):
        """
        filename specified which data split to use (train.csv, dev.csv, test.csv).
        """
        examples = []
        data = []
        # read data
        with open(self.data_path + filename) as json_file:
            data = json.load(json_file)

        i = 1
        # cover answers
        for example in data:
            uid = example['uid']
            question = example['question']
            query = example['query']
            query = self._prepare_query(query)
            examples.append(InputExample(guid=uid, texts=[question, query], label=random.random()))
            i = i+1

        return examples
