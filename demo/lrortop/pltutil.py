from rsf.proj import *
from rsf.recipes import wplot, geom, awe, pplot

def addText(oplt, iplt, NameDict = []):
    if not NameDict:
        NameDict = ["Vp1", 'Vp2', 'Vp3', 'Vn1', 'Vn2', 'Vn3']
    LabelDict = ["V\_pz", 'V\_px', 'V\_py',
                'V\_pz', 'V\_px', 'V\_py']
                #'V\_nmo\s90 1', 'V\_nmo\s90 2', 'V\_nmo\s90 3']
    xcoord = [-0.3, 4.525, 9.05]
    ycoord = [8.1, 4.4]
    vplfile = [iplt]
    for i in range(6):
        x0 = xcoord[i % 3]
        y0 = ycoord[i / 3]
        name = NameDict[i]
        label = LabelDict[i]
        vplfile += [name+'Text']
        Plot(name+'Text', None, '''
            box x0=%g y0=%g label="%s"
                boxit=n pointer=n lab_fat=3 size=0.16
            ''' % (x0, y0, label))
    Plot(oplt, vplfile, 'Overlay')
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
        byte bar=${TARGETS[1]} gainpanel=a pclip=%(pclip)g mean=n bias=2.3
        ''' % pltpar + boundstr)
    plist = []
    for i in range(6):
        basic = wplot.igrey3d('''
                bar=${SOURCES[1]} bartype=h barwidth=0.256 barmove=n
                color=j flat=n point1=0.618 point2=0.618 mean=n bias=2.3
                ''' + boundstr, pltpar)
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
    barX = 10.20
    barY = 0.22
    barUnits = "(km\s75 /\s100 s)"
    Plot(pltname+'_bar_units', None, '''
            box x0=%g y0=%g label="%s"
                boxit=n pointer=n lab_fat=3 size=0.18
            ''' % (barX, barY, barUnits))
    Plot(pltname+'_bar__', pltname+'_bar_', 'Overlay', vppen='ywmin=7.2 yshift=-7.2')
    Plot(pltname+'_bar', [pltname+'_bar__', pltname+'_bar_units'], 'Overlay')
    Plot(pltname+'_byte_yshift', pltname+'_byte', 'Overlay', vppen='ycenter=-1.0')
    pplot.p2x1(pltname+'_noText', pltname+'_byte_yshift', pltname+'_bar', 1, 1, 0.0)
    addText(pltname, pltname+'_noText', [])

def plt3x2_movie(par, nframe, filename, pltname='', pclip=100, flat=False, needBound=False):
    frames = []
    for iframe in range(nframe):
        tag = '%d'%iframe
        frame_name = filename+'_frame_'+tag
        frames += [frame_name]
        Flow(frame_name, filename, 'window n5=1 f5=%d'%iframe)
        plt3x2(par, frame_name, '', pclip, flat, needBound)
    if not pltname:
        pltname = filename+'_movie'
    Result(pltname, frames, 'Movie')    




