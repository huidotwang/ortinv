from rsf.proj import *
from rsf.recipes import wplot, geom, awe
import sys
sys.path.append('../../bin/')
import math as m

bin_mod = '../../bin/mpilrortmod.x'
bin_fwi_dat = '../../bin/mpilrortfwi_dat.x'
mpirun = os.environ['MPIRUN']

def runMod(par, srcWav, sou, rec, mvars, odat, owfl=''):
    par['mpirun'] = mpirun
    par['mod'] = bin_mod
    targets = [odat]
    owfl_str = ' '
    if owfl:
        targets += [owfl] 
        owfl_str += ' snap=y owfl=${TARGETS[1]} '
    Flow(targets, [srcWav, sou, rec, mvars],'''
        %(mpirun)s %(mod)s
        input=${SOURCES[0]} output=${TARGETS[0]}
        sou=${SOURCES[1]} rec=${SOURCES[2]}
        model=${SOURCES[3]}
        verb=%(verb)s nb=%(nb)d
        seed=%(seed)d npk=%(npk)d eps=%(eps)g
        '''%par + owfl_str)

def runFwi(par, srcWav, sou, rec, mvars, dobs, mask, fhi=float('inf'), maxiternum=1):
    par['mpirun'] = mpirun
    par['fwi_dat'] = bin_fwi_dat
    par['maxiternum'] = maxiternum
    deps = [srcWav, dobs, mvars, sou, rec]
    tag = '_lr_full'
    if not m.isinf(fhi):
        tag = '_0To' + str(int(fhi))
        srcWav_ = srcWav + tag
        dobs_ = dobs + tag
        Flow(srcWav_, srcWav, 'transp | bandpass fhi=%d | transp' % fhi)
        Flow(dobs_, dobs, 'transp memsize=30000 | bandpass fhi=%d | transp memsize=30000' % fhi)
        deps = [srcWav_, dobs_, mvars, sou, rec]
    extra = ''
    if mask:
        deps += [mask]
        extra += ' msk=${SOURCES[5]} '
    if 'xlb' in par:
        if par['xlb']:
            extra += ' lb=%(xlb)g ' % par
    if 'xub' in par:
        if par['xub']:
            extra += ' ub=%(xub)g ' % par
    Flow(['dat_X'+tag, 'dat_intermediateX'+tag], deps, '''
        %(mpirun)s %(fwi_dat)s
        input=${SOURCES[0]} output=${TARGETS[0]}
        dat=${SOURCES[1]} model=${SOURCES[2]}
        sou=${SOURCES[3]} rec=${SOURCES[4]}
        log=${TARGETS[1]}
        verb=%(verb)s nb=%(nb)d
        seed=%(seed)d npk=%(npk)d eps=%(eps)g
        maxiternum=%(maxiternum)d
        localdatapath=%(localdatapath)s
        ''' % par + extra, stdin=0, stdout=-1)
