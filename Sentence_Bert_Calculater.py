import args
import logging
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
class SBertSimilarCalculate():
    def __init__(self,model_path=args.retrival_sbert):
        self.model = SentenceTransformer(model_path)

    def predict(self, sentence1, sentence2):
        emb1 = self.model.encode(sentence1,show_progress_bar=False)
        emb2 = self.model.encode(sentence2,show_progress_bar=False)
        cos_sim = util.pytorch_cos_sim(emb1, emb2).numpy()
        return cos_sim[0][0]

class SBertSimCalc():
    def __init__(self,sbert_model):
        if isinstance(sbert_model,SentenceTransformer):
            self._sbert_model = sbert_model
        elif isinstance(sbert_model,str):
            self._sbert_model = SentenceTransformer(sbert_model)
        else:
            raise TypeError('Input Type Error!')
    #@profile
    def predict(self,sentence1,sentence2):
        emb1 = self._sbert_model.encode(sentence1,show_progress_bar=False)
        emb2 = self._sbert_model.encode(sentence2,show_progress_bar=False)
        cos_sim = util.pytorch_cos_sim(emb1, emb2).numpy()
        return cos_sim[0][0]

# Sentences are encoded by calling model.encode()
# emb1 = model.encode("你别给我最高多少了，最低能贷多少啊")
# emb2 = model.encode("你们就没有额度大的贷款了吗")
# cos_sim = util.pytorch_cos_sim(emb1, emb2)
# print("Cosine-Similarity:", cos_sim)

if __name__ == '__main__':
    sbert = SBertSimilarCalculate()
    while(True):
        sentence1 = input('sentence1: ')
        sentence2 = input('sentence2: ')
        print(f'similarity：{sbert.predict(sentence1,sentence2)}')