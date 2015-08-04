# coding: utf-8
# Functions to help write LVQNet to an S-expression file
import os

class SEWriter():
    def __init__(self, name, path):
        # (TODO) add path logic
        ext = '.sexp'
        self.document = open(path + name + ext, 'w')
        self.document.write("(network (name %s)\n" % name)
        
    def writeSExp(self, operator, *args):
        f = self.document
        f.write("(" + operator)
        
        for arg in args:
            f.write(" %s" % str(arg))
        
        f.write(")\n")
        
    def writeSexpArr(self, operator, arr):
        f = self.document
        f.write("(" + operator)
        
        for unit in arr:
            f.write(unit + " ")
        
        f.write(")\n")
    
    def finish(self):
        self.document.write("\n)")
        self.document.close()