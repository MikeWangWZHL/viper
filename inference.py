import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"
os.environ["OPENAI_API_KEY"] = open('api.key').read().strip()
os.environ["CONFIG_NAMES"] = "gpt4"
for name, value in os.environ.items():
    print("{0}: {1}".format(name, value))
from main_simple_lib import *

im = load_image('https://viper.cs.columbia.edu/static/images/kids_muffins.jpg')
query = 'How many muffins can each kid have for it to be fair?'

# show_single_image(im)
code = get_code(query)

import pdb; pdb.set_trace()
try:
    result = execute_code(code, im, show_intermediate_steps=False)
    print(result)
except Exception as e:
    print(e)
    print("Error in executing code")
    result = None

import pdb; pdb.set_trace()