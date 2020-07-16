"""
This script converts the original Excel files to binary format. The import function is
very slow, but need only be run once.
"""
import sys
import pandas as pd
import pickle


def main():
    if len(sys.argv) != 3:
        print("Usage: prog input_file output_file")
        sys.exit(2)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # open with pandas
    print("Reading (this will take a while)...", file=sys.stderr, end=' ', flush=True)
    data = [pd.read_excel(open(input_file, 'rb'), s, 0) for s in ['Defect Data', 'ATA CH-SEC',
                                                                  'MEL Code Data', 'Trax Recurrent Data']]
    print("done.", file=sys.stderr)

    # write binary
    print("Writing...", file=sys.stderr, end=' ', flush=True)
    pickle.dump(data, open(output_file, 'wb'))
    print("done.", file=sys.stderr)


if __name__ == '__main__':
    main()
