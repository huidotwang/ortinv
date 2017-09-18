from rsf.proj import *

lrort = '../../bin/lrortop.x'

# ========================================= #
class lrortop:
    '''lowrank P-wave propagation'''
    def __init__(self, v, ss, rr, par, custom=''):
        self.v = v
        self.custom = custom
        self.par = 'npk=%(npk)d seed=%(seed)d eps=%(eps)g nb=%(nb)d '%par+custom
        
        self.dep = [self.v]
        self.ss = ''
        if (ss != ''):
            self.ss = ' sou=' + ss + '.rsf '
            self.dep.append(ss)
        
        self.rr = ''
        if (rr != ''):
            self.rr = ' rec=' + rr + '.rsf '
            self.dep.append(rr)

    # ------------------------------------- #
    def FORW(self, m, d):
        Flow(d, [m]+self.dep,
            lrort
            + ''' adj=n model=${SOURCES[1]} ''' 
            + self.ss + self.rr + self.par)

    # ------------------------------------- #
    def ADJT(self, m, d):
        Flow(m, [d]+self.dep,
            lrort
            + ''' adj=y model=${SOURCES[1]} ''' 
            + self.ss + self.rr + self.par)
