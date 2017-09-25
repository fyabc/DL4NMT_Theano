import numpy

import cPickle as pkl
import gzip


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class TextIterator:
    """Simple Bitext iterator."""

    def __init__(self, source, target,
                 source_dict, target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1,
                 print_data_file=None,
                 enc_explicit_boundary=None,
                 dec_explicit_boundary=None):
        self.source = fopen(source, 'r')
        self.target = fopen(target, 'r')
        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)
        with open(target_dict, 'rb') as f:
            self.target_dict = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * 40

        self.end_of_data = False
        self.print_data_file = print_data_file

        self.use_enc_explicit_boundary = enc_explicit_boundary is not None
        if self.use_enc_explicit_boundary:
            self.source_boundary = fopen(enc_explicit_boundary, 'r')
            self.source_boundary_buffer = []

        self.use_dec_explicit_boundary = dec_explicit_boundary is not None
        if self.use_dec_explicit_boundary:
            self.target_boundary = fopen(dec_explicit_boundary, 'r')
            self.target_boundary_buffer = []

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)
        if self.use_enc_explicit_boundary:
            self.source_boundary.seek(0)
        if self.use_dec_explicit_boundary:
            self.target_boundary.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        if self.use_enc_explicit_boundary:
            source_boundary = []
        if self.use_dec_explicit_boundary:
            target_boundary = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for k_ in xrange(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                tt = self.target.readline()
                if tt == "":
                    break

                self.source_buffer.append(ss.strip().split())
                self.target_buffer.append(tt.strip().split())

                if self.use_enc_explicit_boundary:
                    s_boundary = self.source_boundary.readline()
                    s_boundary = [float(x) for x in s_boundary.strip().split()]
                    self.source_boundary_buffer.append(s_boundary)

                if self.use_dec_explicit_boundary:
                    t_boundary = self.target_boundary.readline()
                    t_boundary = [float(y) for y in t_boundary.strip().split()]
                    self.target_boundary_buffer.append(t_boundary)

            # sort by target buffer
            tlen = numpy.array([len(t) for t in self.target_buffer])
            tidx = tlen.argsort()

            _sbuf = [self.source_buffer[i] for i in tidx]
            _tbuf = [self.target_buffer[i] for i in tidx]

            self.source_buffer = _sbuf
            self.target_buffer = _tbuf

            if self.use_enc_explicit_boundary:
                _sbbuf = [self.source_boundary_buffer[i] for i in tidx]
                self.source_boundary_buffer = _sbbuf
            if self.use_dec_explicit_boundary:
                _tbbuf = [self.target_boundary_buffer[i] for i in tidx]
                self.target_boundary_buffer = _tbbuf

            if self.print_data_file is not None:
                print "Print raw data!"
                for s, t in zip(self.source_buffer, self.target_buffer):
                    print >>self.print_data_file, s
                    print >>self.print_data_file, t
                    print >>self.print_data_file
                self.print_data_file.close()
                self.print_data_file = None

        if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:
            # actual work here
            while True:
                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                ss = [self.source_dict.get(w, 1)
                      for w in ss]
                if self.n_words_source > 0:
                    ss = [w if w < self.n_words_source else 1 for w in ss]

                # read from target file and map to word index
                tt = self.target_buffer.pop()
                tt = [self.target_dict.get(w, 1)
                      for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]

                if self.use_enc_explicit_boundary:
                    s_boundary = self.source_boundary_buffer.pop()
                if self.use_dec_explicit_boundary:
                    t_boundary = self.target_boundary_buffer.pop()

                if len(ss) > self.maxlen and len(tt) > self.maxlen:
                    continue

                source.append(ss)
                target.append(tt)

                if self.use_enc_explicit_boundary:
                    source_boundary.append(s_boundary)

                if self.use_dec_explicit_boundary:
                    target_boundary.append(t_boundary)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        ret = [source, target]
        if self.use_enc_explicit_boundary:
            ret.append(source_boundary)
        if self.use_dec_explicit_boundary:
            ret.append(target_boundary)

        return ret
