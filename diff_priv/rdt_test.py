import rdt
# print(dir(rdt.transformers.FloatFormatter))

# ['INITIAL_FIT_STATE', 'INPUT_SDTYPE', 'IS_GENERATOR', 
#  'SUPPORTED_SDTYPES', '__class__', '__delattr__', 
#  '__dict__', '__dir__', '__doc__', '__eq__', 
#  '__format__', '__ge__', '__getattribute__', 
#  '__getstate__', '__gt__', '__hash__', '__init__', 
#  '__init_subclass__', '__le__', '__lt__', '__module__', 
#  '__ne__', '__new__', '__reduce__', '__reduce_ex__', 
#  '__repr__', '__setattr__', '__sizeof__', '__str__', 
#  '__subclasshook__', '__weakref__', '_add_columns_to_data', 
#  '_build_output_columns', '_dtype', '_fit', '_get_columns_data', 
#  '_get_output_to_property', '_max_value', '_min_value', '_raise_out_of_bounds_error', 
#  '_reverse_transform', '_rounding_digits', '_set_fitted_parameters', 
#  '_set_missing_value_generation', '_set_model_missing_values', '_set_seed', 
#  '_store_columns', '_transform', '_validate_values_within_bounds', 'column_prefix', 
#  'columns', 'fit', 'fit_transform', 'get_input_column', 'get_input_sdtype', 'get_name', 
#  'get_next_transformers', 'get_output_columns', 'get_output_sdtypes', 'get_subclasses', 
#  'get_supported_sdtypes', 'is_generator', 'missing_value_generation', 
#  'missing_value_replacement', 'model_missing_values', 'null_transformer', 
#  'output_columns', 'random_seed', 'reset_randomization', 'reverse_transform',
# 'set_random_state', 'transform']

ff = rdt.transformers.FloatFormatter() 

print(rdt.transformers.FloatFormatter.__module__)


print(ff._rounding_digits)
print(vars(ff))
