import sklearn_crfsuite
class CRF_baseline_NER():
    def __init__(self):
        pass
    def prepare_features(self):
        pass
    def train(self):
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        crf.fit(self.X_train, self.y_train)
    def save_model(self,path):
        pass
    def predict(self,text):
        pass