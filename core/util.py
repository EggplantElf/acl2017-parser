import time

class Logger:
    def __init__(self, log_file):
        self.log_file = log_file

    def log(self, msg):
        print msg
        with open(self.log_file, 'a') as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S\n", time.localtime()))
            f.write(msg + '\n\n')


class Evaluator:
    def __init__(self):
        self.total = 0
        self.heads = 0
        self.labels = 0


    def evaluate(self, state):
        for d, h, l in state.arcs:
            self.total += 1
            if h == state.sent.gold_heads[d]:
                self.heads += 1
                if l == state.sent.tokens[d].label:
                    self.labels += 1



    def result(self, msg = None):
        uas = 'UAS: %d / %d = %.2f%%' % (self.heads, self.total, self.uas())
        las = 'LAS: %d / %d = %.2f%%' % (self.labels, self.total, self.las())
        s = uas + '\n' + las 
        if msg:
            s = msg + '\n' + s
        return s

    def uas(self):
        return 100.0 * self.heads / self.total

    def las(self):
        return 100.0 * self.labels / self.total


    