#! -*- coding: utf-8 -*-

class results:
    def __init__(self):
        self.d = {}

    def addValue(self,x,y):
        self.d[x] = y


class doer:
    def __init__(self, conveyor):
        self.conveyor = conveyor
    
    def do_something (self,a,b):
        self.conveyor.addValue(a,b)


my_result = results ()
print (my_result.d)

my_doer = doer (my_result)

my_doer.do_something('primero',1)
print (my_result.d)

my_doer.do_something('segundo',2)
print (my_result.d)



