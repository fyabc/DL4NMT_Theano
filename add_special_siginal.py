import argparse

def add_special_siginal(input_file, output_file):
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            words_in = line.strip().split(' ')
            words_out = []
            for word in words_in:
                words_out.append(word)
                if word[-2:] == '@@':
                    words_out.append('-----')
            fout.write(' '.join(words_out) + '\n')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Add special signal to data')

    parser.add_argument('input', help='input filename')
    parser.add_argument('output', help='output filename')

    args = parser.parse_args()
    add_special_siginal(args.input, args.output)