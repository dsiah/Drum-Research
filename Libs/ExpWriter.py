# coding: utf-8
# Functions to help write LVQNet to an S-expression file
import os

class SEWriter():
    def __init__(self, name, path):
        # (TODO) add path logic
        ext = '.sexp'
        self.document = open(path + name + ext, 'w')
        self.document.write("(network (name %s)" % name)
        
    def writeSExp(self, operator, *args):
        f = self.document
        f.write("\n\t(%s"   % str(operator))
        
        for arg in args:
            f.write(" %s" % str(arg))
        
        f.write(")")
        
    def writeSexpArr(self, operator, arr):
        f = self.document
        f.write("\n\t(%s"   % str(operator))
        
        for unit in arr:
            f.write(" %s" % str(unit))
        
        f.write(")")
    
    def finish(self):
        self.document.write(")")
        self.document.close()