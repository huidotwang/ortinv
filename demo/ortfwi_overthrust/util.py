from rsf.proj import *
from rsf.recipes import wplot
import math as m

#----------------------------------------
def gauss3d(gaus,xcen,ycen,zcen,xsig,ysig,zsig,par):
    Flow(gaus,None,
    '''
    math output="exp( -(x1-(%g))^2/(2*%g) -(x2-(%g))^2/(2*%g) -(x3-(%g))^2/(2*%g) )"
    ''' % (zcen,zsig*zsig,xcen,xsig*xsig,ycen,ysig*ysig) +
    '''
    n1=%(nz)d d1=%(dz)g o1=%(oz)g
    n2=%(nx)d d2=%(dx)g o2=%(ox)g 
    n3=%(ny)d d3=%(dy)g o3=%(oy)g |
    scale axis=123
    ''' % par)
#----------------------------------------
def horizontal3d(cc, zcoord, par, nx, ny, jx=1, jy=1, fx=0, fy=0):
    import random, os
    def add(x,y): return x+y
    def myid(n): return '_'+reduce(add,['%d'%random.randint(0,9) for i in range(n)])
    M8R='$RSFROOT/bin/sf'
    DPT=os.environ.get('TMPDATAPATH',os.environ.get('DATAPATH'))

    cco=cc+'o'+myid(16)
    ccz=cc+'z'+myid(16)
    ccy=cc+'y'+myid(16)
    ccx=cc+'x'+myid(16)

    # nx=(par['nx']-fx-1)/jx+1
    # ny=(par['ny']-fy-1)/jy+1

    Flow(cc,None,
    '''
    %smath output=0 n1=%d o1=%g d1=%g n2=%d o2=%g d2=%g |
    window f1=%d j1=%d n1=%d f2=%d j2=%d n2=%d >%s datapath=%s/;
    '''%(M8R,
    par['nx'],par['ox'],par['dx'],
    par['ny'],par['oy'],par['dy'],
    fx,jx,nx,fy,jy,ny,cco,DPT) +
    '''
    %smath <%s output="%g" | put n2=1 n1=%d >%s datapath=%s/;
    '''%(M8R,cco,zcoord,nx*ny,ccz,DPT) +
    '''
    %smath <%s output="x1" | put n2=1 n1=%d >%s datapath=%s/;
    '''%(M8R,cco,nx*ny,ccx,DPT) +
    '''
    %smath <%s output="x2" | put n2=1 n1=%d >%s datapath=%s/;
    '''%(M8R,cco,nx*ny,ccy,DPT) +
    '''
    %scat axis=2 space=n %s %s %s | 
    transp | 
    put o1=0 d1=1 o2=0 d2=1 label1="" unit1="" label2="" unit2="">${TARGETS[0]};
    '''%(M8R,ccx,ccy,ccz) +
    '''     
    %srm %s %s %s %s
    '''%(M8R,cco,ccx,ccy,ccz),
    stdin=0,
    stdout=0)

#----------------------------------------
# -------------
class srplot2d:
    '''extract 2d information from 3d parameter sets'''
    def __init__(self, par):
        self.par2d = dict( 
                    lz=par['lx'], uz=par['ux'], zmin=par['ox'], 
                    lx=par['ly'], ux=par['uy'], xmin=par['oy'],
                    zmax=par['ox']+par['dx']*(par['nx']-1), 
                    xmax=par['oy']+par['dy']*(par['ny']-1) )
        self.par2d['iratio2d']= 1.0 * par['dx'] / par['dy']
        if (self.par2d['iratio2d']>=0.8):  self.par2d['iheight2d']=10
        else:  self.par2d['iheight2d']=12*self.par2d['iratio2d']
        self.par2d['labelattr']=' parallel2=n labelsz=7 labelfat=4 titlesz=12 titlefat=3 xll=2.5 yll=1. '

    def cgraph2d(self, custom):
        return '''
        graph title=""
        labelrot=n wantaxis=n yreverse=y wherexlabel=t
        min2=%g max2=%g label2=%s unit2=%s
        min1=%g max1=%g label1=%s unit1=%s
        screenratio=%g screenht=%g wantscalebar=n
        %s
        ''' % ( self.par2d['zmin'],self.par2d['zmax'],self.par2d['lz'],self.par2d['uz'],
                self.par2d['xmin'],self.par2d['xmax'],self.par2d['lx'],self.par2d['ux'],
                self.par2d['iratio2d'],self.par2d['iheight2d'],
                self.par2d['labelattr']+' '+custom)

    def plot2d(self, custom):
        return '''window n1=2 | dd type=complex | ''' + self.cgraph2d('symbol=. plotcol=3 plotfat=5'+' '+custom)
# -----------------------------------------------------------------


# ===============================
def unitPowerBall(ballName, xcen, ycen, zcen,
                     xradius, yradius, zradius,
                     par, needplot, flatplot=False):
    ''' Pwoerball with Gaussian shape, unit maximum magnitude'''
    wplot.param(par) 
    centers = [
        [m.sqrt(3.0), 1.0], [0.0, 2.0], [-m.sqrt(3.0), 1.0],
        [-m.sqrt(3.0), -1.0], [0.0, -2.0], [m.sqrt(3.0), -1.0]]
    centers = map(lambda (x,y): [xcen + x * xradius, ycen + y * xradius], centers)
    xsig = 0.512 * xradius
    ysig = 0.512 * yradius
    zsig = 0.512 * zradius
    ball_list = []
    for i in range(6):
        xcoord = centers[i][0]
        ycoord = centers[i][1]
        zcoord = zcen
        name = ballName +'_%d'%i
        ball_list += [name]
        gauss3d(name, xcoord, ycoord, zcoord, xsig, ysig, zsig, par)
        xframe = xcoord/par['dx']
        yframe = ycoord/par['dy']
        if needplot == True:
            plotstring = wplot.igrey3d('flat=n point1=0.618 point2=0.618 frame2=%d frame3=%d color=j '%(xframe,yframe),par)
            if flatplot: plotstring += ' flat=y '
            Result(name, 'byte pclip=100 gainpanel=a | '+ plotstring)
            # --------- cat plots -----------
            extra = ' wantaxis1=n wantaxis2=n wantaxis3=n '
            if i == 0 or i == 3: extra += ' wantaxis1=y '
            if i == 2 or i == 5: extra += ' wantaxis3=y '
            if i > 2 and i < 6:  extra += ' wantaxis2=y '
            Plot(name, 'byte pclip=100 gainpanel=a | '+ plotstring+extra)
    from rsf.recipes import pplot
    pplot.p1x3(ballName+'_merge_up',ball_list[0],ball_list[1],ball_list[2],1,1,-9.10)
    pplot.p1x3(ballName+'_merge_down',ball_list[3],ball_list[4],ball_list[5],1,1,-9.10)
    pplot.p2x3(ballName+'_merge',
               ball_list[0], ball_list[1], ball_list[2],
               ball_list[3], ball_list[4], ball_list[5],
               0.5,0.5,-6.1,-9.1)
                

    Flow(ballName+'_all', ball_list, 'add ${SOURCES[1:-1]}')
    if needplot == True:
        yframe = centers[5][1] / par['dy']
        plotstring = wplot.igrey3d('flat=n point1=0.618 point2=0.618 frame3=%d color=j '%yframe,par)
        # ratio = (par['nz']*par['dz']+par['oz']) / (par['nx']*par['dx']+par['ox'])
        # heit = 12.0 * ratio
        # plotstring += ' screenratio=%f screenht=%f '%(ratio, heit)
        if flatplot: plotstring += ' flat=y '
        Result(ballName+'_all', 'byte pclip=100 gainpanel=a mean=y | '+plotstring)
    
    if needplot == True:
        yframe = centers[5][1] / par['dy']
        plotstring = wplot.igrey3d('flat=n color=i frame3=%d'%yframe,par)
        if flatplot: plotstring += ' flat=y '
        Plot(ballName+'_all', 'byte pclip=100 gainpanel=a mean=y | '+plotstring)
        Plot('box_source_zSlice_scaled', 'box_source_zSlice','Overlay', vppen='yscale=0.6 xscale=0.6 xcenter=-1.652')
        pplot.p2x1(ballName+'_all_source',
              ballName+'_all' , 'box_source_zSlice_scaled',
               1,1,-1.518)



# ===============================
def boxSources(srcName, par, xcen, ycen, zcoord, xradius, yradius, jx=1, jy=1, needplot=False):
    '''units are all in grid samples'''
    fx = xcen - xradius * jx 
    fy = ycen - yradius * jy
    nx = 2 * xradius + 1
    ny = 2 * yradius + 1
    horizontal3d(srcName, zcoord, par, nx, ny, jx, jy, fx, fy)

    if needplot:
        splot = srplot2d(par)
        Result(srcName+'_zSlice', srcName, splot.plot2d(''))
        Plot(srcName+'_zSlice', srcName, splot.plot2d('plotfat=20 plotcol=5'))
    
# ===============================
def boxReceivers(recName, par):
    ''' five faces receivers surrounding the cube'''
    from rsf.recipes import geom
    xlo_slice = recName+'_xlo'
    xfi_slice = recName+'_xfi'
    ylo_slice = recName+'_ylo'
    yfi_slice = recName+'_yfi'
    zlo_slice = recName+'_zlo'
    zfi_slice = recName+'_zfi'
    xlo_coord = par['ox']
    xfi_coord = par['ox']+(par['nx']-1)*par['dx']
    ylo_coord = par['oy']
    yfi_coord = par['oy']+(par['ny']-1)*par['dy']
    zlo_coord = par['oz']
    zfi_coord = par['oz']+(par['nz']-1)*par['dz']

    geom.YZsheet3d(xlo_slice, xlo_coord, '', par, jy=1, jz=1)
    geom.YZsheet3d(xfi_slice, xfi_coord, '', par, jy=1, jz=1)
    geom.ZXsheet3d(ylo_slice, ylo_coord, '', par, jx=1, jz=1)
    geom.ZXsheet3d(yfi_slice, yfi_coord, '', par, jx=1, jz=1)
    geom.XYsheet3d(zlo_slice, zlo_coord, '', par, jx=1, jy=1)
    geom.XYsheet3d(zfi_slice, zfi_coord, '', par, jx=1, jy=1)

    Flow(recName,[xlo_slice, xfi_slice, ylo_slice, yfi_slice, zfi_slice, zlo_slice], 'cat axis=2 ${SOURCES[1:-1]}')


