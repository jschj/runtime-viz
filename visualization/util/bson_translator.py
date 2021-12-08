import json
import bson


def tranlate_json_to_bson(inputfilepath, outputfilepath):
    with open(inputfilepath, 'r') as inputfile:
        content = json.load(inputfile)

    binary = bson.dumps(content)

    with open(outputfilepath, 'wb') as outputfile:
        outputfile.write(binary)


def dump_bson(inputfilepath):
    with open(inputfilepath, 'rb') as inputfile:
        binary = inputfile.read()
        content = bson.loads(binary)

    print(content)
