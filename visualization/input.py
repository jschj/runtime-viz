from data.buffer import Buffer1D, Buffer2D
import bson


def read_input(inputfilepath: str):
    content = {}

    with open(inputfilepath, 'rb') as inputfile:
        binary = inputfile.read()
        content = bson.loads(binary)

    buffers = {}

    for bufferdetails in content["buffers"]:
        id = bufferdetails["id"]
        type = bufferdetails["type"]

        buffer = {}
        if type == "plain":
            buffer = Buffer1D(bufferdetails)
        elif type == "pitched":
            buffer = Buffer2D(bufferdetails)
        else:
            raise "Invalid buffer type!"

        buffers[id] = buffer

    return buffers
