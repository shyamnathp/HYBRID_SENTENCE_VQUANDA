"""VerbalDataset"""
import os
import re
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchtext.data import Field, Example, Dataset

from utils.constants import (
    ANSWER_TOKEN, ENTITY_TOKEN, SOS_TOKEN, EOS_TOKEN,
    SRC_NAME, TRG_NAME, TRAIN_PATH, TEST_PATH
)
class VerbalDataset(object):
    """VerbalDataset class"""
    TOKENIZE_SEQ = lambda self, x: x.replace("?", " ?").\
                                     replace(".", " .").\
                                     replace(",", " ,").\
                                     replace("'", " '").\
                                     split()
    ANSWER_REGEX = r'\[.*?\]'
    QUERY_DICT = {
        'x': 'var_x',
        'uri': 'var_uri',
        '{': 'brack_open',
        '}': 'brack_close',
        '.': 'sep_dot',
        'COUNT(uri)': 'count_uri'
    }
    ROOT_PATH = Path(os.path.dirname(__file__))
    def __init__(self):
        self.train_path = str(self.ROOT_PATH.parent.parent) + TRAIN_PATH
        self.test_path = str(self.ROOT_PATH.parent.parent) + TEST_PATH
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.src_field = None
        self.trg_field = None
        self.earl_entities = self._read_earl_entites(str(self.ROOT_PATH) + '/earl_entities.json')
        with open('/data/premnadh/Hybrid-QASystem/LCQuad/train-data.json') as json_fileTwo:
            self.dataLCQTrain = json.load(json_fileTwo)
        with open('/data/premnadh/Hybrid-QASystem/LCQuad/test-data.json') as json_fileThree:
            self.dataLCQTest = json.load(json_fileThree)
        self.templateJson = json.load(open('/data/premnadh/Hybrid-QASystem/Utils/templates.json'))      

    def _read_earl_entites(self, path):
        entities = []
        with open(path) as json_file:
            entities = json.load(json_file)
        keys = [ent['uid'] for ent in entities]
        entities = dict(zip(keys, entities))
        return entities

    def _cover_answers(self, text):
        """
        Cover answers on text using an answer token
        """
        return re.sub(self.ANSWER_REGEX, ANSWER_TOKEN, text)

    def _cover_entities(self, uid, question, answer):
        """
        Cover entities on a given text.
        Since we use external entity recognizer it might
        miss some entities or cover wrong ones.
        A better approach will be to annotate all data
        and cover all entities. This should improve model
        performance.
        """
        # Try EARL for covering entities
        # EARL results are serialized
        data = self.earl_entities[uid]
        question_entities = data['question_entities']
        answer_entities = data['answer_entities']
        # we cover all recognized entitries by the same token
        # this has to be improved based on the number of entities.
        # For example if we have 2 entities we create 2 tokens e.g. <ent1> <ent2>
        # In this way we know the position of each entity in the translated output
        for ent in question_entities: question = question.replace(ent, ENTITY_TOKEN)
        for ent in answer_entities: answer = answer.replace(ent, ENTITY_TOKEN)

        return question, answer

    def _prepare_query(self, query, uid, cover_entities):
        """
        trasnform query from this:
        SELECT DISTINCT COUNT(?uri) WHERE { ?x <http://dbpedia.org/ontology/commander> <http://dbpedia.org/resource/Andrew_Jackson> . ?uri <http://dbpedia.org/ontology/knownFor> ?x  . }
        To this:
        select distinct count_uri where brack_open var_x commander Andrew Jackson sep_dot var_uri known for var_x sep_dot brack_close
        """
        lcTrain = [val for val in self.dataLCQTrain if val['_id'] == uid]
        lcTest = [val for val in self.dataLCQTest if val['_id'] == uid]
        tempId = None
        if(lcTrain):
            tempId = (lcTrain[0])['sparql_template_id']
        elif(lcTest):
            tempId = (lcTest[0])['sparql_template_id']

        tempEntry = [val for val in self.templateJson if val['id'] == tempId]
        #if(len(tempEntry) != 0):
        logical_form = (tempEntry[0])['logical_form']

        logical_form = logical_form.replace('(', ' ( ')\
                                   .replace(')', ' ) ')\
                                   .replace(',', ' ,').strip()

        query = query.replace('\n', ' ')\
                     .replace('\t', '')\
                     .replace('?', '')\
                     .replace('{?', '{ ?')\
                     .replace('>}', '> }')\
                     .replace('{uri', '{ uri')\
                     .replace('uri}', 'uri }').strip()
        query = query.split()
        new_query = []
        i = j = 1
        classType = 0
        for q in query:
            if q in self.QUERY_DICT:
                q = self.QUERY_DICT[q]
            if 'http' in q:
                if 'dbpedia.org/ontology' in q or 'dbpedia.org/property' in q:
                    original_q = q
                    original_q = original_q.replace('<','').replace('>','')
                    q = q.rsplit('/', 1)[-1].lstrip('<').rstrip('>')
                    q = filter(None, re.split("([A-Z][^A-Z]*)", q))
                    q = ' '.join(q)

                    with open('/data/premnadh/Hybrid-QASystem/Utils/predicates.txt', 'r') as file1:
                        contents = file1.readlines()
                        nContents = [con.replace(',\n', '') for con in contents]
                        if original_q in nContents:
                            predicate = 'pred'+str(j)
                            logical_form = logical_form.replace(predicate, q.lower())
                            j += 1

                    if(classType == 1):
                        logical_form = logical_form.replace('class', q.lower())
                        classType = 0
                elif 'www.w3.org/1999/02/22-rdf-syntax-ns#type' in q:
                    q = 'type'
                    classType = 1
                elif cover_entities:
                    q = ENTITY_TOKEN + str(i)
                    i += 1
                else:
                    q = q.rsplit('/', 1)[-1].lstrip('<').rstrip('>').replace('_', ' ')
            new_query.append(q.lower())

        assert new_query[-1] == 'brack_close', 'Query not ending with a bracket.'
        return ' '.join(new_query), logical_form

    def _extract_question_answer(self, train, test):
        return [[data['question'], data['verbalized_answer']] for data in train], \
                [[data['question'], data['verbalized_answer']] for data in test]

    def _extract_query_answer(self, train, test):
        return [[data['query'], data['verbalized_answer']] for data in train], \
                [[data['query'], data['verbalized_answer']] for data in test]

    def _make_torchtext_dataset(self, data, fields):
        examples = [Example.fromlist(i, fields) for i in data]
        return Dataset(examples, fields)

    def load_data_and_fields(self, cover_entities=False, query_as_input=False):
        """
        Load verbalization data
        Create source and target fields
        """
        train, test, val = [], [], []
        # read data
        with open(self.train_path) as json_file:
            train = json.load(json_file)

        with open(self.test_path) as json_file:
            test = json.load(json_file)

        # cover answers
        for data in train: data.update((k, self._cover_answers(v)) for k, v in data.items() if k == "verbalized_answer")
        for data in test: data.update((k, self._cover_answers(v)) for k, v in data.items() if k == "verbalized_answer")

        # cover entities or prepare query if query is used
        if cover_entities or query_as_input:
            for data in [train, test]:
                for example in data:
                    uid = example['uid']
                    question = example['question']
                    answer = example['verbalized_answer']
                    query = example['query']
                    if cover_entities: 
                        question, answer = self._cover_entities(uid, question, answer)
                    if query_as_input: 
                        query, logical_form = self._prepare_query(query, uid, cover_entities)
                        example.update(question=question, verbalized_answer=answer, query=logical_form)
                    else:
                        example.update(question=question, verbalized_answer=answer, query=query)
                    


        # extract question-answer or query-answer pairs
        if query_as_input:
            train, test = self._extract_query_answer(train, test)
        else:
            train, test = self._extract_question_answer(train, test)

        # split test data to val-test
        test, val = train_test_split(test, test_size=0.5, shuffle=False)

        # create fields
        self.src_field = Field(tokenize=self.TOKENIZE_SEQ,
                               init_token=SOS_TOKEN,
                               eos_token=EOS_TOKEN,
                               lower=True,
                               include_lengths=True,
                               batch_first=True)
        self.trg_field = Field(tokenize=self.TOKENIZE_SEQ,
                               init_token=SOS_TOKEN,
                               eos_token=EOS_TOKEN,
                               lower=True,
                               batch_first=True)

        fields_tuple = [(SRC_NAME, self.src_field), (TRG_NAME, self.trg_field)]

        # create toechtext datasets
        self.train_data = self._make_torchtext_dataset(train, fields_tuple)
        self.valid_data = self._make_torchtext_dataset(val, fields_tuple)
        self.test_data = self._make_torchtext_dataset(test, fields_tuple)

        # build vocabularies
        self.src_field.build_vocab(self.train_data, min_freq=2)
        self.trg_field.build_vocab(self.train_data, min_freq=2)

    def get_data(self):
        """Return train, validation and test data objects"""
        return self.train_data, self.valid_data, self.test_data

    def get_fields(self):
        """Return source and target field objects"""
        return self.src_field, self.trg_field

    def get_vocabs(self):
        """Return source and target vocabularies"""
        return self.src_field.vocab, self.trg_field.vocab
