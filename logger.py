import os


class Logger():
    def __init__(self, opt):
        self.traintxt = open(os.path.join(opt.checkpoint_path, 'trainlog.txt'), 'a')

    def write(self, log, printlog=True):
        if printlog:
            print(log)
        self.traintxt.write(log+'\n')

    def write_dict(self, dict):
        for key, value in dict.iteritems():
            temp = str(key) + ':' + str(value) + '\n'
            self.traintxt.write(temp)



