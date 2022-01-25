from argparse import ArgumentParser


files = get_files("../input/freesound-audio-ds-project/ProcessedAudio/baby cry")

create_dataset("dataset", files[:1])    # TODO: files[x: y] -> files



def set_args():
    parser = ArgumentParser()
    parser.add_argument('arg1', type=str)
    parser.add_argument('arg2', type=str)
    parser.add_argument('arg3', type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    print(args)
    with open('input.txt', 'r') as input_:
        content = input_.readlines()
        with open('output.txt', 'w') as output:
            output.write(content)
