# -*- coding: utf-8 -*-
__author__ = 'AlexInTown'


def save_submissions(fname, ids, preds):
    assert len(ids) == len(preds), "Error: the id and pred length not match!"
    f = open(fname, 'w')
    f.write("ID,TARGET\n")
    for i in xrange(len(ids)):
        f.write("{0},{1}\n".format(ids[i], preds[i]))
    f.close()
    pass

