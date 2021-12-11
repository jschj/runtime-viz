import json

import bson

import buffer


def read_input(inputfilepath: str) -> buffer.BufferCollection:
    buffers = {}

    # read input file
    if inputfilepath.lower().endswith(".bson"):
        with open(inputfilepath, 'rb') as inputfile:
            binary = inputfile.read()
            content = bson.loads(binary)
    elif inputfilepath.lower().endswith(".json"):
        with open(inputfilepath, 'r') as inputfile:
            text = inputfile.read()
            content = json.loads(text)
    else:
        raise "Input file has an invalid file extension!"

    # initialize buffers
    for bufferdetails in content["buffers"]:
        identifier = bufferdetails["id"]
        buffertype = bufferdetails["type"]

        if buffertype == "plain":
            b = buffer.Buffer1D(bufferdetails)
        elif buffertype == "pitched":
            b = buffer.Buffer2D(bufferdetails)
        else:
            raise "Invalid buffer type!"

        buffers[identifier] = b

    for accessdetails in content["accesses"]:
        bufferid = accessdetails["b"]

        if bufferid in buffers:
            buffers[bufferid].add_memory_access(accessdetails)
        else:
            raise "Found a memory access for a buffer which does not exist!"

    for _, b in buffers.items():
        b.sanity_checks()

    return buffers
