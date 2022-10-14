import json
import os

class InputDocument:
    """Represents a document that consists of sentences and a label for each sentence"""
    
    def __init__(self, sentences, labels, doc_name):
        """sentences: array of sentences labels: array of labels for each sentence """
        self.sentences = sentences
        self.labels = labels
        self.doc_name = doc_name

    def get_sentence_count(self):
        return len(self.sentences)



class DocumentsDataset:
    def __init__(self, path, max_docs=-1):
        self.path = path    
        self.length = None
        self.max_docs = max_docs
    
    #Adapter functions for Iterator 
    def __iter__(self):
        return self.readfile()
    
    def __len__(self):
        return self.calculate_len()    
    
    
    def calculate_len(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length
    
    def readfile(self):
        """Yields InputDocuments """
        read_docs = 0
        with open(self.path, encoding="utf-8") as f:
            sentences, tags = [], []
            doc_name=''
            for line in f:
                if self.max_docs >= 0 and read_docs >= self.max_docs:
                    return
                line = line.strip()
                if not line:
                    if len(sentences) != 0:
                        read_docs += 1
                        yield InputDocument(sentences, tags,doc_name)
                        sentences, tags = [], []
                        doc_name = ''
                elif not line.startswith("###"):
                    ls = line.split("\t")
                    if len(ls) < 2:
                        continue
                    else:
                        tag, sentence = ls[0], ls[1]
                    sentences += [sentence]
                    tags += [tag]

                elif line.startswith("###"):
                    doc_name = line.replace("###","").strip()

