import json
import bson


def tranlate_json_to_bson(inputfilepath, outputfilepath):
    content = {}

    with open(inputfilepath, 'r') as inputfile:
        content = json.load(inputfile)

    binary = bson.dumps(content)

    with open(outputfilepath, 'wb') as outputfile:
        outputfile.write(binary)


def dump_bson(inputfilepath):
    content = {}
    with open(inputfilepath, 'rb') as inputfile:
        binary = inputfile.read()
        content = bson.loads(binary)

    print(content)
