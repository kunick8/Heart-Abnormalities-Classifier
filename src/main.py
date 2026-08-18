from src.data.data_preprocessing import DataPreprocessing
from src.data.data_cleaning import join_data
from src.data.data_extractor import extract_data

dataset1 = extract_data('SPECTF.test')
dataset2 = extract_data('SPECTF.train')
dataset = join_data(dataset1, dataset2)


Preprocessor = DataPreprocessing(dataset)
preprocessed_data = Preprocessor.get_preprocessed_data()


