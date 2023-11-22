import pandas as pd
from joblib import dump
from simpletransformers.ner import NERModel, NERArgs
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class NERModelTrainer:
    def __init__(self, dataset_path, model_type='bert', model_name='bert-base-cased'):
        self.dataset_path = dataset_path
        self.model_type = model_type
        self.model_name = model_name
        self.model = None

    def load_data(self):
        data = pd.read_csv(self.dataset_path, encoding="latin1")
        data = data.fillna(method="ffill")

        label_encoder = LabelEncoder()
        data["Sentence #"] = label_encoder.fit_transform(data["Sentence #"])
        data.rename(columns={"Sentence #": "sentence_id", "Word": "words", "Tag": "labels"}, inplace=True)
        data["labels"] = data["labels"].str.upper()

        return data

    def prepare_data(self, data):
        X = data[["sentence_id", "words"]]
        Y = data["labels"]

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

        train_data = pd.DataFrame({"sentence_id": x_train["sentence_id"], "words": x_train["words"], "labels": y_train})
        test_data = pd.DataFrame({"sentence_id": x_test["sentence_id"], "words": x_test["words"], "labels": y_test})

        return train_data, test_data

    def train(self, train_data, test_data):
        label_list = train_data["labels"].unique().tolist()

        args = NERArgs()
        args.num_train_epochs = 1
        args.learning_rate = 1e-4
        args.overwrite_output_dir = True
        args.train_batch_size = 32
        args.eval_batch_size = 32

        self.model = NERModel(self.model_type, self.model_name, labels=label_list, args=args)
        self.model.train_model(train_data, eval_data=test_data, acc=accuracy_score)

    def evaluate(self, test_data):
        result, model_outputs, preds_list = self.model.eval_model(test_data)
        return result

    def predict(self, text):
        prediction, model_output = self.model.predict([text])
        return prediction


if __name__ == "__main__":
    ner_trainer = NERModelTrainer(r'ner_dataset.csv')
    data = ner_trainer.load_data()
    train_data, test_data = ner_trainer.prepare_data(data)
    ner_trainer.train(train_data, test_data)
    dump(ner_trainer, 'model.joblib')
