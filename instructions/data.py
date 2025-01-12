import json
import os
import urllib
import urllib.request

def downlaod_and_load_file(filepath, url):
    if not os.path.exists(filepath):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(filepath, 'w', encoding='utf-8') as fp:
            fp.write(text_data)

    with open(filepath, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data

def format_input_alpaca(entry):
    return f"""Below is an instruction that describes a task.
Write a response that appropriately completes the request."""+\
f"""\n\n### Instruction:
{entry['instruction']}"""+(f"""\n\n### Input:
{entry['input']}""" if entry['input'] else '')