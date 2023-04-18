from src.features.build_features import ExtractLetter, ExtractTitle
from src.config.core import config


def test_extract_letter_transformer(sample_input_data):
    expected_first_transformer = str
    expected_13th_transformer = '?'
    expected_last_transformer = str

    transformer = ExtractLetter(variable=config.model_config.var_to_extract_letter)
    subject = transformer.fit_transform(sample_input_data)

    assert expected_first_transformer == type(subject[config.model_config.var_to_extract_letter].iloc[0])
    assert expected_13th_transformer == subject[config.model_config.var_to_extract_letter].iloc[12]
    assert expected_last_transformer == type(subject[config.model_config.var_to_extract_letter].iloc[-1])


def test_extract_title_transformer(sample_input_data):
    expected_first_transformer = 'Mrs'
    expected_13th_transformer = 'Mr'
    expected_last_transformer = 'Mr'

    transformer = ExtractTitle(variable=config.model_config.var_to_extract_title)
    subject = transformer.fit_transform(sample_input_data)

    assert expected_first_transformer == subject[config.model_config.title_var_name].iloc[0]
    assert expected_13th_transformer == subject[config.model_config.title_var_name].iloc[12]
    assert expected_last_transformer == subject[config.model_config.title_var_name].iloc[-1]

