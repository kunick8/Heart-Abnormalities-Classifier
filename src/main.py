from src.data.data_preprocessing import DataPreprocessing
from src.data.data_cleaning import join_data
from src.data.data_extractor import extract_data
from src.model import Classifier

dataset1 = extract_data('../data/raw/SPECTF.test')
dataset2 = extract_data('../data/raw/SPECTF.train')
dataset = join_data(dataset1, dataset2)


Preprocessor = DataPreprocessing(dataset)
preprocessed_data = Preprocessor.get_preprocessed_data()
classifier = Classifier('SVC')
model = classifier.create_best_model()

model.fit(preprocessed_data[0], preprocessed_data[2])