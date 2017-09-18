from rsf.proj import *

class modelBuilder:
    def __init__(self, par, vpzFileName):
        self.par = par
        self.vpz = vpzFileName
        self.msk = self.vpz + '_msk'
        # Flow(self.msk, self.vpz, 'mask min=1.52 max=4.6 | dd type=float')
        Flow(self.msk, self.vpz, 'mask min=1.52 max=4.3 | dd type=float | smooth rect1=3 rect2=3 rect3=3 repeat=3 | math output="abs(input)"')
    
    def run(self, nameDict):
        vpz = self.vpz
        msk = self.msk
        eps1 = nameDict['eps1']
        eps2 = nameDict['eps2']
        del1 = nameDict['del1']
        del2 = nameDict['del2']
        del3 = nameDict['del3']
        vpx = nameDict['vpx']
        vpy = nameDict['vpy']
        vn1 = nameDict['vnmo1']
        vn2 = nameDict['vnmo2']
        vn3 = nameDict['vnmo3']
        vpa1 = nameDict['vpa1']
        vpa2 = nameDict['vpa2']
        vpa3 = nameDict['vpa3']
        Flow(eps1, [vpz, msk], 'math msk=${SOURCES[1]} output="input/167.5*msk"')
        Flow(eps2, [vpz, msk], 'math msk=${SOURCES[1]} output="input/168.5*msk"')
        Flow(del1, [vpz, msk], 'math msk=${SOURCES[1]} output="input/169.5*msk"')
        Flow(del2, [vpz, msk], 'math msk=${SOURCES[1]} output="input/170.5*msk"')
        Flow(del3, [vpz, msk], 'math msk=${SOURCES[1]} output="input/171.5*msk"')
        Flow(vpx, [vpz, eps2], 'math v=${SOURCES[0]} e=${SOURCES[1]} output="v*sqrt(1+2*e)"')
        Flow(vpy, [vpz, eps1], 'math v=${SOURCES[0]} e=${SOURCES[1]} output="v*sqrt(1+2*e)"')
        Flow(vn1, [vpz, del1], 'math v=${SOURCES[0]} d=${SOURCES[1]} output="v*sqrt(1+2*d)"')
        Flow(vn2, [vpz, del2], 'math v=${SOURCES[0]} d=${SOURCES[1]} output="v*sqrt(1+2*d)"')
        Flow(vn3, [vpx, del3], 'math v=${SOURCES[0]} d=${SOURCES[1]} output="v*sqrt(1+2*d)"')
        Flow(vpa1, [vn1, vpz], 'math pn=${SOURCES[0]} p=${SOURCES[1]} output="sqrt(pn * p)"')
        Flow(vpa2, [vn2, vpz], 'math pn=${SOURCES[0]} p=${SOURCES[1]} output="sqrt(pn * p)"')
        Flow(vpa3, [vn3, vpx], 'math pn=${SOURCES[0]} p=${SOURCES[1]} output="sqrt(pn * p)"')
