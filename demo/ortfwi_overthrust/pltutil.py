from rsf.proj import *
from rsf.recipes import wplot, geom, awe, pplot

def addText(oplt, iplt, NameDict = []):
    if not NameDict:
        NameDict = ['Vp1', 'Vp2', 'Vp3', 'Vn1', 'Vn2', 'Vn3']
    xcoord = [0.0, 4.525, 9.05]
    ycoord = [8.3, 4.6]
    vplfile = [iplt]
    for i in range(6):
        x0 = xcoord[i % 3]
        y0 = ycoord[i / 3]
        name = NameDict[i]
        vplfile += [name+'Text']
        Plot(name+'Text', None, '''
            box x0=%g y0=%g label=%s
                boxit=n pointer=n lab_fat=3 size=0.16
            ''' % (x0, y0, name))
    Result(oplt, vplfile, 'Overlay')


def plt3x2(par, filename, pltname='', pclip=100, flat=False, needBound=False):
    from copy import deepcopy
    pltpar = deepcopy(par)
    wplot.param(pltpar)
    pltpar['pclip'] = pclip
    boundstr = ''
    if needBound:
        boundstr = ' minval=%(xlb)g maxval=%(xub)g ' % pltpar
    if not pltname:
        pltname = filename
    bytefile = filename + '_byte'
    barfile = filename + '_bar'
    Flow([bytefile, barfile], filename, 
        '''
        byte bar=${TARGETS[1]} mean=y gainpanel=a pclip=%(pclip)g
        ''' % pltpar + boundstr)
    plist = []
    for i in range(6):
        basic = wplot.igrey3d('''
                bar=${SOURCES[1]} bartype=h barwidth=0.256 barmove=n
                color=j flat=n point1=0.618 point2=0.618 mean=y
                ''', pltpar)
        if flat: pltstr += ' flat=y '
        extra = ' wantaxis1=n wantaxis2=n wantaxis3=n '
        if i == 0 or i == 3: extra += ' wantaxis1=y '
        if i == 2 or i == 5: extra += ' wantaxis3=y '
        if i > 2 and i < 6:  extra += ' wantaxis2=y '
        pltstr = basic + extra
        plist += [pltname+'_%d'%i]
        Plot(pltname+'_%d'%i, [bytefile, barfile],'''
            window n4=1 f4=%d | '''%i + pltstr)
    pplot.p2x3(pltname+'_byte',
              plist[0], plist[1],  plist[2],
              plist[3], plist[4],  plist[5],
              0.5, 0.5, -7.4, -9.2)
    Plot(pltname+'_bar_',[bytefile, barfile],'''
        window n4=1 |''' + pltstr + ' scalebar=y')
    Plot(pltname+'_bar', pltname+'_bar_', 'Overlay', vppen='ywmin=7.2 yshift=-7.2')
    Plot(pltname+'_byte_yshift', pltname+'_byte', 'Overlay', vppen='ycenter=-1.0')
    pplot.p2x1(pltname+'_noText', pltname+'_byte_yshift', pltname+'_bar', 1, 1, 0.0)
    addText(pltname, pltname+'_noText', [])

