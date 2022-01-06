import json
import bson


def tranlate_json_to_bson(inputfilepath, outputfilepath):
    with open(inputfilepath, 'r') as inputfile:
        content = json.load(inputfile)

    binary = bson.dumps(content)

    with open(outputfilepath, 'wb') as outputfile:
        outputfile.write(binary)


def tranlate_bson_to_json(inputfilepath, outputfilepath):
    with open(inputfilepath, 'rb') as inputfile:
        binary = inputfile.read()
        content = bson.loads(binary)

    text = json.dumps(content, indent=2)

    with open(outputfilepath, 'w') as outputfile:
        outputfile.write(text)


def dump_bson(inputfilepath):
    with open(inputfilepath, 'rb') as inputfile:
        binary = inputfile.read()
        content = bson.loads(binary)

    print(content)
