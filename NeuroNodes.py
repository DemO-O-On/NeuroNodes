import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"           #       Прятать hello pygame и тд
import pygame as pg
import time
import traceback
import pickle

pg.display.set_icon(pg.image.load('Icon.ico'))

'''t1 = time.process_time()
import keras
print(time.process_time()-t1)'''

f = open('pas.txt')
f = f.read()
f = f.split('---')
f = f[1:]

h = open('arg.txt')
h = h.read()
h = h.split('---')[1:]

zag = []

arr = pg.image.load('Arrow.png')
arr = pg.transform.scale(arr, (30, 30))

par = {}

for i in f:
    i = i.split('\n')
    i = i[1:-1]
    i = list(map(lambda a: (a[0].upper() + a[1:]).replace('_', ' '), i))
    zag.append(i)

for i in h:
    i = i.split('\n')
    if i[1].find('; ') != -1:
        i[1] = i[1].split('; ')
        #i[1] = list(map(lambda a: (a[0].upper() + a[1:]).replace('_', ' '), i[1]))
    else:
        i[1] = [i[1]]
    '''if i[0].find('Model') == -1:
        i[1].append('name: ')'''
    par[i[0]] = i[1]



cv = {
'Model': (255, 164, 161),
'Layer activations': (246,85,119),
'Layer weight initializers': (215, 148, 190),
'Layer weight regularizers': (0, 132, 75),
'Layer weight constraints': (111, 35, 205),
'Core layers': (245, 197, 80),
'Convolution layers': (140, 195, 75),
'Pooling layers': (0, 202, 202),
'Recurrent layers': (53, 92, 181),
'Preprocessing layers': (255, 123, 90),
'Normalization layers': (175, 198, 236),
'Regularization layers': (145, 104, 191),
'Attention layers': (190, 157, 140),
'Reshaping layers': (205, 92, 92),
'Merging layers': (64, 152, 179),
'Locally-connected layers': (228, 204, 41),
'Activation layers': (134, 198, 152)
}



pg.font.init()



sizex, sizey = 1200, 800
window = pg.display.init()
window = pg.display.set_mode((sizex, sizey),pg.RESIZABLE)        
pg.display.set_caption("NeuroNodes")

sbl = 200

sc = pg.Surface((sizex, sizey-sbl))
sc2 = pg.Surface((sizex-250, sizey-sbl))
bl = pg.Surface((sizex, sbl))

rasst = 50

clock = pg.time.Clock()
p1 = 0
rig = []
rab = []    #   Main list of layers
rabl = []
smx, smy = 0, 0
fl2 = False
fl3 = False
fl4 = False
fl5 = False
fls = False
iz = False
iz2vr = []
iz3 = False
vr = False
skm = False
sm = False
activt = False
dl = False
raz = False
blact = False
flrm = False
flzakr = False
flas = False
blof = []
ctrl = False
curn = 0
col = 0
cur = 0
curs = 0
cursf = []
smt = 0
has = 0
blot = ''
model = ''
ts = ''
ind = ''
curf = []
nncu = []
smf = []
text = []
text2 = []
chosed = []
rich = [0 for i in range(17)]

blot = '''

#-----Here model is training-----


'''



cur = 0


massh = 1
fon = pg.font.Font('UbuntuMono-Regular.ttf', int(15/massh))
fons = pg.font.Font('UbuntuMono-Regular.ttf', 36)

import copy
pg.scrap.init()


def rot_center(image, angle):
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    rot_image = rot_image.subsurface(rot_rect).copy()
    return rot_image


class A:
    def __init__(self, x, y, sx, sy, tip, ist, perpar, *dv):
        self.perpar = copy.deepcopy(perpar)
        self.params = []
        self.a = pg.Rect(x, y, sx, sy)
        self.rg = pg.Rect(self.a.bottomright[0], self.a.topleft[1], 10, self.a.size[1])
        self.lf = pg.Rect(self.a.topleft[0]-10, self.a.topleft[1], 10, self.a.size[1])
        self.tip = tip
        self.ist = ist
        self.persv = []
        self.zadsv = []
        self.ncrr = 1
        self.ncrl = 1
        self.br = 'X'
        self.l = False
        if dv:
            self.dv = True
        else:
            global has
            self.has = has
            has += 1
            self.dv = False



    def create(self, ncrl, ncrr):
        global cv
        global text
        global text2
        global massh
        self.rg = pg.Rect(self.a.bottomright[0]-3, self.a.topleft[1], 6, self.a.size[1])
        self.lf = pg.Rect(self.a.topleft[0]-3, self.a.topleft[1], 6, self.a.size[1])
        self.params = []
        self.circr = []
        self.circl = []
        self.ncrr = ncrr
        self.ncrl = ncrl

        if self.dv:
            #   (63,63,63)
            self.osn = pg.draw.rect(sc2, pg.Color(154,154,154), (self.a.topleft[0], self.a.topleft[1]+20, self.a.size[0], self.a.size[1]-22), border_bottom_left_radius=7, border_bottom_right_radius = 7)
            self.sh = pg.draw.rect(sc2, cv[self.tip], (self.a.topleft[0], self.a.topleft[1], self.a.size[0], 22), border_top_left_radius = 7, border_top_right_radius = 7)
            self.osn = pg.draw.rect(sc, pg.Color(154,154,154), (self.a.topleft[0], self.a.topleft[1]+20, self.a.size[0], self.a.size[1]-22), border_bottom_left_radius=7, border_bottom_right_radius = 7)
            self.sh = pg.draw.rect(sc, cv[self.tip], (self.a.topleft[0], self.a.topleft[1], self.a.size[0], 22), border_top_left_radius = 7, border_top_right_radius = 7)

            self.t = fon.render(self.ist, True, (0, 0, 0))
            sc.blit(self.t, (self.a.topleft[0] + self.a.size[0]//2 - self.t.get_size()[0]//2, self.a.topleft[1] + 3))
            sc2.blit(self.t, (self.a.topleft[0] + self.a.size[0]//2 - self.t.get_size()[0]//2, self.a.topleft[1] + 3))


            #   (161,161,161)

            for i in range(self.ncrr):
                self.circr.append(pg.draw.circle(sc2, (0,20,0), (self.a.topleft[0]+self.a.size[0]-1, self.a.topleft[1] + 10 + self.a.size[1]/(ncrr+1)*(i+1)), 5))

            for i in range(self.ncrr):
                pg.draw.circle(sc, (0,20,0), (self.a.topleft[0]+self.a.size[0]-1, self.a.topleft[1] + 10 + self.a.size[1]/(ncrr+1)*(i+1)), 5)


            if self.ist != 'Input object':
                for i in range(self.ncrl):
                    self.circl.append(pg.draw.circle(sc2, (0,20,0), (self.a.topleft[0]+1, self.a.topleft[1] + 10 + self.a.size[1]/(ncrl+1)*(i+1)), 5))

                for i in range(self.ncrl):
                    pg.draw.circle(sc, (0,20,0), (self.a.topleft[0]+1, self.a.topleft[1] + 10 + self.a.size[1]/(ncrl+1)*(i+1)), 5)




            for i in range(len(self.perpar)):
                if len(self.perpar[i])*8 > self.a.size[0]-20:
                    if self.perpar[i].find('"') != -1:
                        self.params.append([self.perpar[i][:int((self.a.size[0]-20)/8)-1], pg.Rect(self.a.topleft[0]+10, self.a.topleft[1]+23*(i+1) + (self.a.size[1] - 20 - len(self.perpar)*23)//2, self.a.size[0]-20, 20)])
                    else:
                        self.params.append([self.perpar[i][:int((self.a.size[0]-20)/8)], pg.Rect(self.a.topleft[0]+10, self.a.topleft[1]+23*(i+1) + (self.a.size[1] - 20 - len(self.perpar)*23)//2, self.a.size[0]-20, 20)])
                else:
                    self.params.append([self.perpar[i], pg.Rect(self.a.topleft[0]+10, self.a.topleft[1]+23*(i+1) + (self.a.size[1] - 20 - len(self.perpar)*23)//2, self.a.size[0]-20, 20)])

                #   (0,100,0)
                pg.draw.rect(sc2, pg.Color(125,193,125), (self.params[-1][1].topleft[0], self.params[-1][1].topleft[1], self.params[-1][1].size[0], self.params[-1][1].size[1]), border_radius=7)
                pg.draw.rect(sc, pg.Color(125,193,125), (self.params[-1][1].topleft[0], self.params[-1][1].topleft[1], self.params[-1][1].size[0], self.params[-1][1].size[1]), border_radius=7)
                p = fon.render(self.params[-1][0], True, (0, 0, 0))
                sc.blit(p, (self.params[-1][1].topleft[0]+5, self.params[-1][1].topleft[1]+3))
                sc2.blit(p, (self.params[-1][1].topleft[0]+5, self.params[-1][1].topleft[1]+3))


        else:
            self.osn = pg.draw.rect(sc2, pg.Color(154,154,154), (self.a.topleft[0], self.a.topleft[1]+20, self.a.size[0], self.a.size[1]-22), border_bottom_left_radius=7, border_bottom_right_radius = 7)
            self.sh = pg.draw.rect(sc2, cv[self.tip], (self.a.topleft[0], self.a.topleft[1], self.a.size[0], 22), border_top_left_radius = 7, border_top_right_radius = 7)

            self.t = fon.render(self.ist, True, (0, 0, 0))
            sc2.blit(self.t, (self.a.topleft[0] + self.a.size[0]//2 - self.t.get_size()[0]//2, self.a.topleft[1] + 3))

            for i in range(self.ncrr):
                self.circr.append(pg.draw.circle(sc2, (0,20,0), (self.a.topleft[0]+self.a.size[0]-1, self.a.topleft[1] + 10 + self.a.size[1]/(ncrr+1)*(i+1)), 5))
            if self.ist != 'Input object':
                for i in range(self.ncrl):
                    self.circl.append(pg.draw.circle(sc2, (0,20,0), (self.a.topleft[0]+1, self.a.topleft[1] + 10 + self.a.size[1]/(ncrl+1)*(i+1)), 5))



            for i in range(len(self.perpar)):
                if len(self.perpar[i])*8 > self.a.size[0]-25:
                    if self.perpar[i].find('"') != -1:
                        self.params.append([self.perpar[i][:int((self.a.size[0]-25)/8)-1], pg.Rect(self.a.topleft[0]+10, self.a.topleft[1]+23*(i+1) + (self.a.size[1] - 20 - len(self.perpar)*23)//2, self.a.size[0]-20, 20)])
                    else:
                        self.params.append([self.perpar[i][:int((self.a.size[0]-25)/8)], pg.Rect(self.a.topleft[0]+10, self.a.topleft[1]+23*(i+1) + (self.a.size[1] - 20 - len(self.perpar)*23)//2, self.a.size[0]-20, 20)])
                else:
                    self.params.append([self.perpar[i], pg.Rect(self.a.topleft[0]+10, self.a.topleft[1]+23*(i+1) + (self.a.size[1] - 20 - len(self.perpar)*23)//2, self.a.size[0]-20, 20)])
                

                pg.draw.rect(sc2, pg.Color(125,193,125), (self.params[-1][1].topleft[0], self.params[-1][1].topleft[1], self.params[-1][1].size[0], self.params[-1][1].size[1]), border_radius=7)
                p = fon.render(self.params[-1][0], True, (0, 0, 0))
                sc2.blit(p, (self.params[-1][1].topleft[0]+5, self.params[-1][1].topleft[1]+3))





    def move(self,x,y):
        self.a.topleft = (x,y)

    def check(self):
        return self.a.collidepoint(p1)

    def check_text(self):
        for i in range(len(self.params)):
            if self.params[i][1].collidepoint(p1):
                return i
            
    def check_gr(self):
        if self.lf.collidepoint(p1):
            return 'l'
        elif self.rg.collidepoint(p1):
            return 'r'
        else:
            return False


class ZG:
    def __init__(self, x, y, sx, sy, m):
        self.a = pg.Rect(x, y, sx, sy)
        self.nod = m
        self.sm = 25
        self.kn = []

    def create(self, t, ch):
        self.ch = ch
        global cv
        self.b = pg.draw.rect(sc, cv[t], (self.a.topleft[0], self.a.topleft[1], self.a.size[0], self.a.size[1]), border_radius=7)

        global arr
        self.arr = arr
    
        if self.ch == 1:
            self.arr = pg.transform.rotate(arr, -90)
        sc.blit(self.arr, (self.a.topleft[0] + self.a.size[0]-25, self.a.topleft[1] - 5))

        
        self.text = t
        if len(self.text)*9 > self.a.size[0]:
            self.t = fon.render(t[:self.a.size[0]//9-1], True, (0, 0, 0))
        else:
            self.t = fon.render(t, True, (0, 0, 0))
        sc.blit(self.t, (self.a.topleft[0] + 6, self.a.topleft[1] + 3))
        #text.append([self.t, self.a.topleft[0] + self.a.size[0]//2 - self.t.get_size()[0]//2, self.a.topleft[1] + 3])


    def razv(self):
        it = 0
        self.mal = []
        global text
        for i in self.nod:

            self.kn.append(pg.Rect(self.a.topleft[0] + 10, self.a.topleft[1] + 25 + it*self.sm, self.a.size[0], self.a.size[1]))

            b = pg.draw.rect(sc, pg.Color(111, 103, 204), (self.kn[-1].topleft[0], self.kn[-1].topleft[1], self.kn[-1].size[0], self.kn[-1].size[1]), border_radius=7)

            it = it + 1
            self.mal.append(i)


            if len(i)*9 > self.a.size[0]:
                t = fon.render(i[:self.a.size[0]//9-1], True, (255, 255, 255))
            else:
                t = fon.render(i, True, (255, 255, 255))
                
            sc.blit(t, (self.kn[-1].topleft[0] + 6, self.kn[-1].topleft[1] + 3))

    def check(self):
        return self.a.collidepoint(p1)

    def check_kn(self):
        for i in self.kn:
            i.collidepoint(p1)
            if i.collidepoint(p1):
                return i



class Bloc:
    def __init__(self):
        global sbl
        global ws
        global blot
        self.a = pg.Rect(0, ws[1]-sbl, ws[0], sbl)
        self.t = blot

    def draw(self, sm):
        global cur
        global blot
        c = 0
        a = cur
        ci = 0
        o = 0
        if self.t == '\x08':
            self.t = ''
            cur = 0
            a = 0
            blot = ''
        for i in self.t.split('\n'):
            i = i.replace('\x08', '')

            if o != 'A':
                if a > len(i):
                    a -= len(i)+1
                else:
                    ci = c
                    o = 'A'

            p = fon.render(i, True, (255, 255, 255))
            bl.blit(p, (20, 10+c*17+15*sm))
            c = c + 1



        if o != 'A':
            pg.draw.rect(bl, (255,0,0), (20+len(self.t.split('\n')[-1])*8, 12+(c-1)*17+15*sm, 2, 13), width=1)
        else:
            pg.draw.rect(bl, (255,0,0), (20+a*8, 12+ci*17+15*sm, 2, 13), width=1)

        

        
    def check_bl(self):
        return self.a.collidepoint(p1)


kom = 0
inp = []
flv = False
poso = (360, 140)

mos = 'A'


def vglub(b):
    global inp
    global model
    global flv
    global mos
    a = []
    pred = []
    for i in b:
        if i not in pred:
            pred.append(i)
            co = 0
            for j in i:
                if len(j.persv) > 0:
                    a.append(list(map(lambda a: a[1], j.persv)))



            if len(i) > 1:
                for j in i:
                    pp = ''
                    for g in j.perpar:
                        pp += g.replace(':', ' =') + ', '
                    pp = pp[:-2]
                    j.br = j.br + '_' + str(co)

                    
                    if j.zadsv[0][1].ist == 'Input object':
                        model += j.br + ' = keras.layers.' + j.ist.split(' ')[0].strip().replace(':', ' =') + '(' + pp + ')(X_input)\n\n'
                        flv = False

                    else:
                        if len(j.zadsv) > 1:
                            for g in j.zadsv:
                                model += j.br + ' = keras.layers.' + j.ist.split(' ')[0].strip().replace(':', ' =') + '(' + pp + ')(' + g[1].br + ')\n\n'
                    co += 1



            else:
                try:
                    if i[0].ist != 'Input object':
                        i[0].br = i[0].zadsv[0][1].br
                except IndexError as e:
                    flo = True
                    print('-------------')
                    print('No Input layer')
                    print('-------------')

                try:
                    if i[0].ist == 'Model':
                        if mos == 'A':
                            ll = ''
                            for g in i[0].zadsv:
                                ll += g[1].br + ', '
                            ll = ll[:-2]
                            mos = 'model = keras.models.Model(' + str(mo[0]).replace("'", '').replace(':', ' =') + ', outputs = [' + ll.replace(':', ' =') + '], ' + str(mo[2]).replace(':', ' =') + ')\n\n'


                    elif i[0].tip == 'Model':
                        pp = ''
                        for j in i[0].perpar:
                            pp += j.replace(':', ' =') + ', '
                        pp = pp[:-2]
                        if mos != 'A':
                            model += mos
                            mos = 'A'
                        model += 'model.' + str(i[0].ist.lower()) + '(' + pp + ')\n\n'


                        
                    elif i[0].ist == 'Input object':
                        model += 'X_input = keras.layers.' + i[0].ist.split(' ')[0].strip().replace(':', ' =') + '(' + str(i[0].perpar)[2:-2].replace("'", '').replace(':', ' =') + ')\n\n'
                        flv = True

                        
                    else:
                        pp = ''
                        for j in i[0].perpar:
                            pp += j.replace(':', ' =') + ', '
                        pp = pp[:-2]
                        if flv:
                            model += i[0].br + ' = keras.layers.' + i[0].ist.split(' ')[0].strip().replace(':', ' =') + '(' + pp + ')(X_input)\n\n'
                            flv = False
                        else:
                            for g in i[0].zadsv:
                                model += i[0].br + ' = keras.layers.' + i[0].ist.split(' ')[0].strip().replace(':', ' =') + '(' + pp + ')(' + g[1].br + ')\n\n'
                except IndexError as e:
                    pass
    if a == []:
        inp = []
        model += mos
        return False

    vglub(a)


    

def an():
    global rab
    for i in rab:
        if i[0].ist == 'Input object':
            inp.append(i[0])

ws = pg.display.get_window_size()
lefgr = ws[0] - 250






while 1:
    rio = []
    ws = pg.display.get_window_size()
    text = []
    text2 = []
    it = 0
    pg.draw.aaline(sc, (255,255,255), (lefgr,0), (lefgr,ws[1]))

    fk = pg.Rect((ws[0]-250)//2-30, 0, 60, 30)
    pg.draw.rect(sc2, (66,152,79), fk, border_bottom_left_radius = 7, border_bottom_right_radius = 7)
    p = fon.render('RUN', True, (255, 255, 255))
    sc2.blit(p, (fk.topleft[0]+20, fk.topleft[1]+7))



    scb = pg.Rect(0, 10, 100, 30)
    pg.draw.rect(sc2, (45,150,150), scb, border_bottom_right_radius = 7, border_top_right_radius = 7)
    p = fon.render('Screenshot', True, (255, 255, 255))
    sc2.blit(p, (scb.topleft[0]+10, scb.topleft[1]+7))


    razm = pg.Rect(0, ws[1]-sbl-3, ws[0], 6)
    pg.draw.rect(sc2, (0,152,0), razm)
    pg.draw.rect(sc, (0,152,0), razm)


    zap = pg.Rect(0, 50, 60, 30)
    pg.draw.rect(sc2, (45,150,150), zap, border_bottom_right_radius = 7, border_top_right_radius = 7)
    p = fon.render('Save', True, (255, 255, 255))
    sc2.blit(p, (zap.topleft[0]+12, zap.topleft[1]+7))


    op = pg.Rect(0, 90, 60, 30)
    pg.draw.rect(sc2, (45,150,150), op, border_bottom_right_radius = 7, border_top_right_radius = 7)
    p = fon.render('Open', True, (255, 255, 255))
    sc2.blit(p, (op.topleft[0]+12, op.topleft[1]+7))

    

    if flzakr == False:
        zakr = pg.Rect(lefgr, (ws[1]-sbl)//2-10, 10, 20)
        pg.draw.rect(sc, (0,0,100), zakr, border_top_right_radius = 3, border_bottom_right_radius = 3)
        pg.draw.rect(sc2, (0,0,100), zakr, border_top_right_radius = 3, border_bottom_right_radius = 3)
    else:
        zakr = pg.Rect(lefgr-10, (ws[1]-sbl)//2-10, 10, 20)
        pg.draw.rect(sc2, (0,0,100), zakr, border_top_left_radius = 3, border_bottom_left_radius = 3)
        pg.draw.rect(sc, (0,0,100), zakr, border_top_left_radius = 3, border_bottom_left_radius = 3)

    

    rio.append(ZG(lefgr+30, 50 + smt, 190, 20, zag[0][1:]))
    rio[-1].create(zag[0][0], rich[0])
    blo = Bloc()
    

    for i in range(len(zag[1:])):
        i = i + 1
        if rich[i-1] == 1:
            rio.append(ZG(lefgr+30, 50 + rio[i-1].a.topleft[1] + len(rio[i-1].nod)*rio[0].sm, 190, 20, zag[i][1:]))
            rio[-1].create(zag[i][0], rich[i])
        else:
            rio.append(ZG(lefgr+30, 50 + rio[i-1].a.topleft[1], 190, 20, zag[i][1:]))
            rio[-1].create(zag[i][0], rich[i])


    for i in range(len(rich)):
        if rich[i] == 1:
            rio[i].razv()




    
    
    
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            raise SystemExit
        if event.type == pg.WINDOWMOVED:
            poso = (event.x, event.y)
            
            
        if pg.mouse.get_pos() != p1:
            p1 = pg.mouse.get_pos()


        if p1[0] > lefgr:         #       Все типы нодов
            if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                pos_cl = pg.mouse.get_pos()
                for i in range(len(rio)):
                    if rio[i].check():
                        if rich[i] == 0:
                            rich[i] = 1
                        else:
                            rich[i] = 0



            
        if event.type == pg.MOUSEBUTTONDOWN and p1[0] > lefgr:        #       Создание перетаскиванием
            pos_cl = pg.mouse.get_pos()
            for i in rio:
                if i.check_kn():

                    smx = i.kn[i.kn.index(i.check_kn())].topleft[0] - pos_cl[0]
                    smy = i.kn[i.kn.index(i.check_kn())].topleft[1] - pos_cl[1]
                    #fl2 = A(p1[0]+smx, p1[1]+smy, 200, len(par[i.mal[i.kn.index(i.check_kn())]])*25+30, i.text, i.mal[i.kn.index(i.check_kn())], par[i.mal[i.kn.index(i.check_kn())]], True)

                    #print(par[i.mal[i.kn.index(i.check_kn())]])
                    fl2 = A(p1[0]+smx, p1[1]+smy, 300, len(par[i.mal[i.kn.index(i.check_kn())]])*25+30, i.text, i.mal[i.kn.index(i.check_kn())], copy.deepcopy(par[i.mal[i.kn.index(i.check_kn())]]), True)
                    fl4 = False
                    chosed = False
                    break


        if event.type == pg.MOUSEWHEEL and p1[0] > lefgr:
            smt += event.y*25


        if event.type == pg.MOUSEBUTTONDOWN and p1[1] < ws[1] - sbl:        #       Перемещение всех на скм
            if fl2 == False:
                if event.button == 2:
                    skm = p1




        if event.type == pg.MOUSEBUTTONDOWN:        #       Парсинг модели
            if fk.collidepoint(p1):
                print('Process started')
                model = ''
                an()
                vglub([inp])
                ou = open('Model.py', 'w')
                ou.write(blot[:blot.find('#-----Here model is training-----')]+'\n')
                ou.write(model+'\n')
                ou.write(blot[blot.find('#-----Here model is training-----'):])
                ou.close()
                del ou
                try:
                    exec(open('Model.py').read())
                except ValueError as e:
                    print('-------------')
                    print(e)
                    print('-------------')
                except NameError as e:
                    print('-------------')
                    print(e)
                    print('-------------')
                except ConnectionResetError as e:
                    print('-------------')
                    print(e)
                    print('-------------')
                except AttributeError as e:
                    print('-------------')
                    print(e)
                    print('-------------')
                except ModuleNotFoundError as e:
                    print('-------------')
                    print(e)
                    print('-------------')
                except KeyboardInterrupt as e:
                    print('-------------')
                    print(e)
                    print('-------------')
                except IndexError as e:
                    print('-------------')
                    print(e)
                    print('-------------')
                except SyntaxError as e:
                    print('-------------')
                    print(e)
                    print('-------------')
                print('Process finished')
                model = ''
                for i in rab:
                    i[0].br = 'X'



        if event.type == pg.MOUSEBUTTONDOWN:        #       Сохранение нодов
            if zap.collidepoint(p1):
                fls = True
                ind = 'Zap'




                
        if fls == False and ts != '' and ind == 'Zap':
            ind = ''
            with open(ts, 'wb') as ff:
                for i in rab:
                    ab = [[[int(i[0].a.topleft[0]), int(i[0].a.topleft[1]), int(i[0].a.size[0]), int(i[0].a.size[1]), str(i[0].tip), str(i[0].ist), copy.deepcopy(i[0].perpar), str(i[0].br), i[0].has]]]
                    lll = []
                    for j in i[0].persv:
                        lll.append(j[3])
                    rrr = []
                    for j in i[0].zadsv:
                        rrr.append(j[3])

                    ab[-1].append(lll)
                    ab[-1].append(rrr)

                    pickle.dump(ab, ff)

                pickle.dump(blot, ff)

                if len(rab) > 0:
                    del lll
                    del rrr
                    del ab
                    del ff
                ts = ''
                curs = 0
                print('Nodes saved')








#--------------------------------------------------------------------------------------------------------------------------------











        if event.type == pg.MOUSEBUTTONDOWN:        #       Открытие нодов
            if op.collidepoint(p1):
                fls = True
                ind = 'Op'




                
        if fls == False and ts != '' and ind == 'Op':
            ind = ''
            try:
                ff = open(ts, 'rb')
                print('Nodes opened')
                ts = ''
                curs = 0
                frab = []
                mhas = {}
                fff = []
                while True:
                    try:
                        fff.append(pickle.load(ff))
                    except EOFError:
                        break

            
                del ff
                blot = fff[-1]
                cur = len(blot)
                fff = fff[:-1]
                for i in fff:
                    i = i[0]
                    frab.append([A(i[0][0], i[0][1], i[0][2], i[0][3], i[0][4], i[0][5], i[0][6]), 1, 1])
                    frab[-1][0].br = i[0][-2]
                    frab[-1][0].has = i[0][-1]
                    
                    mhas[frab[-1][0].has] = frab[-1][0]

                    frab[-1][0].circr = []
                    frab[-1][0].circl = []
                    
                    frab[-1][0].persv = []
                    frab[-1][0].zadsv = []
                    
                    frab[-1][0].circr.append(pg.draw.circle(sc2, (0,20,0), (frab[-1][0].a.topleft[0]+frab[-1][0].a.size[0]-1, frab[-1][0].a.topleft[1] + 10 + frab[-1][0].a.size[1]/(1+1)), 5))
                    if frab[-1][0].ist != 'Input object':
                        frab[-1][0].circl.append(pg.draw.circle(sc2, (0,20,0), (frab[-1][0].a.topleft[0]+1, frab[-1][0].a.topleft[1] + 10 + frab[-1][0].a.size[1]/(1+1)), 5))


                k = 0

                for i in fff:
                    i = i[0]
                
                    for g in i[1]:
                        frab[k][0].persv.append([0, mhas[g], 0, g])
                        mhas[g].zadsv.append([0, frab[k][0], 0, frab[k][0].has])
                    

                    k += 1

                vrhas = []
                for i in frab:
                    
                    rab.append([i[0], [1, 1]])
                    vrhas.append(rab[-1][0].has)

                    has = max(vrhas)+1
                del vrhas
                    

            except FileNotFoundError:
                curs = 0
                ts = ''
                print('File not found')


#--------------------------------------------------------------------------------------------------------------------------------


        if p1[0] > lefgr:                                           #       Вспомогательный при скм
            if event.type == pg.MOUSEBUTTONUP and fl2:
                fl2 = False



                    
        if p1[0] < lefgr and p1[1] < ws[1] - sbl:             #       Добавление в рабы
            if fl2 != False:
                if event.type == pg.MOUSEBUTTONUP:
                    rab.append([A(p1[0] + smx, p1[1] + smy, fl2.a.size[0], fl2.a.size[1], fl2.tip, fl2.ist, fl2.perpar), [fl2.ncrl, fl2.ncrr]])
                    fl2 = False

                    

        if p1[0] < lefgr and p1[1] < ws[1] - sbl:
            if event.type == pg.MOUSEBUTTONDOWN:            #           Линии от кружочков
                if p1[0] < lefgr:
                    for i in range(len(rab)):
                        for j in range(len(rab[i][0].circr)):
                            if p1[0] > rab[i][0].circr[j].topleft[0] and p1[0] < rab[i][0].circr[j].bottomright[0] and p1[1] > rab[i][0].circr[j].topleft[1] and p1[1] < rab[i][0].circr[j].bottomright[1]:
                                iz = rab[i][0].circr[j]
                                iz2vr = [i,j]
                                break

                            

        if p1[0] < lefgr and p1[1] < ws[1] - sbl:
            if iz != False:                                 #           Связи между кружочками
                if event.type == pg.MOUSEBUTTONUP:
                    for i in range(len(rab)):
                        for j in range(len(rab[i][0].circl)):
                            if p1[0] > rab[i][0].circl[j].topleft[0] and p1[0] < rab[i][0].circl[j].bottomright[0] and p1[1] > rab[i][0].circl[j].topleft[1] and p1[1] < rab[i][0].circl[j].bottomright[1]:
                                if vr:
                                    vr[1].persv.append([vr[2], rab[i][0], vr[0], rab[i][0].has])      # От кого, кому и куда
                                    rab[i][0].zadsv.append([vr[2], vr[1], vr[0], vr[1].has])

                                    print(rab[i][0].zadsv[-1])
                                    print(vr[1].persv[-1])
                                    
                                    vr = False
                                    break
                                
                                rab[iz2vr[0]][0].persv.append([iz2vr[1], rab[i][0], j, rab[i][0].has])
                                rab[i][0].zadsv.append([j, rab[iz2vr[0]][0], iz2vr[1], rab[iz2vr[0]][0].has])

                                iz2vr = []
                                break







        if p1[0] < lefgr and p1[1] < ws[1] - sbl:       #       При отсоединении линии от кружочка
                    if event.type == pg.MOUSEBUTTONDOWN:        
                        for i in range(len(rab)):
                            for j in range(len(rab[i][0].persv)):
                                pom = rab[i][0].persv[j][1].circl
                                pom = pom[rab[i][0].persv[j][2]]
                                if p1[0] > pom.topleft[0] and p1[0] < pom.bottomright[0] and p1[1] > pom.topleft[1] and p1[1] < pom.bottomright[1]:
                                    
                                    vr = [rab[i][0].persv[j][0], rab[i][0], rab[i][0].persv[j][2], rab[i][0].has]
                                    
                                    iz = rab[i][0].circr[rab[i][0].persv[j][0]]

                                    rab[i][0].persv[j][1].zadsv.remove([rab[i][0].persv[j][2], rab[i][0], rab[i][0].persv[j][0], rab[i][0].has])

                                    rab[i][0].persv.pop(j)
                                    break
                            





        if p1[0] < lefgr and p1[1] < ws[1] - sbl:       #       Вспомогательный при именении размера
            if event.type == pg.MOUSEBUTTONDOWN and event.button == 1 and iz == False and vr == False:
                for i in range(len(rab)):
                    if rab[i][0].check_gr():
                        if rab[i][0].check_gr() == 'l':
                            raz = [i, p1, 'l']
                        elif rab[i][0].check_gr() == 'r':
                            raz = [i, p1, 'r']
                        chosed = False
                        fl4 = False
                        fl3 = False
                        break
                            
                    


        


        
        if p1[0] < lefgr and p1[1] < ws[1] - sbl:
            if event.type == pg.MOUSEBUTTONDOWN:            #       Выбранные
                if fl4 == False:
                    for i in rab:
                        for j in i[0].circl:                #       Чтобы при касании левых кружочков не становился выбранным
                            if p1[0] > j.topleft[0] and p1[0] < j.bottomright[0] and p1[1] > j.topleft[1] and p1[1] < j.bottomright[1]:
                                fl5 = j

                        if fl5:
                            continue
                        
                        if i[0].check():
                            if len(iz2vr) != 0:
                                if rab[iz2vr[0]][0] == i[0]:
                                    continue
                            chosed = i
                            fl4 = True

                if fl4 == False:
                    chosed = False





        
        if p1[0] < lefgr and p1[1] < ws[1] - sbl:
            if event.type == pg.MOUSEBUTTONDOWN:            #       Перетаскивание
                for i in rab:
                    if iz or fl5:     #                                   Для того, чтобы при нажатии на кружочек не перетаскивалась сама нода
                        break
                    if i[0].check():
                        pos_cl = pg.mouse.get_pos()
                        smx = i[0].a.topleft[0] - pos_cl[0]
                        smy = i[0].a.topleft[1] - pos_cl[1]
                        fl3 = i





        if p1[0] < lefgr and p1[1] < ws[1] - sbl:
            if event.type == pg.KEYDOWN:            #           Удаление выбранного
                if event.key == 127:
                    '''if activt:
                        if rab[activt[0]] == chosed[0]:
                            activt = False'''
                    if chosed:
                        for i in rab:
                            for j in i[0].persv:
                                if j[1] == chosed[0]:
                                    i[0].persv.remove(j)
                        for i in rab:
                            for j in i[0].zadsv:
                                if j[1] == chosed[0]:
                                    i[0].zadsv.remove(j)
                        chosed[0].persv.clear()
                        chosed[0].zadsv.clear()
                        rab.remove(chosed)
                        chosed = False




        if p1[0] < lefgr and p1[1] < ws[1] - sbl:       #       Выбирание параметра в ноде для ввода текста
            if event.type == pg.MOUSEBUTTONDOWN:
                activt = False
                nncu = []
                for i in range(len(rab)-1, -1, -1):
                    if rab[i][0].check_text() != None:
                        if chosed and fl4:
                            if chosed[0] == rab[i][0]:
                                curn = len(rab[i][0].perpar[rab[i][0].check_text()])
                                activt = [rab[i][0], rab[i][0].check_text()]
                                chosed = False
                                fl4 = False
                                fl3 = False
                                break





        if razm.collidepoint(p1) and event.type == pg.MOUSEBUTTONDOWN:
            flrm = True

        if zakr.collidepoint(p1) and event.type == pg.MOUSEBUTTONDOWN:
            if flzakr == False:
                lefgr = ws[0]
                flzakr = True
            else:
                lefgr = ws[0]-250
                flzakr = False
            


        if fls:
            if event.type == pg.KEYDOWN:
                if event.key == 27:
                    fls = False
                    ts = ''
                    curs = 0



        if activt and p1[1] < ws[1] - sbl:          #       Ввод текста в параметры
            if event.type == pg.KEYDOWN:
                if event.key == 8 and len(activt[0].perpar[activt[1]]) != 1:
                    if len(activt[0].perpar[activt[1]]) != 0 and activt[0].perpar[activt[1]][curn-2:curn] != ': ':
                        curn -= 1
                        activt[0].perpar[activt[1]] = activt[0].perpar[activt[1]][:curn] + activt[0].perpar[activt[1]][curn+1:]
                        nncu = ['del', time.time()]
                elif event.key == 13:
                    activt = False
                    curn = 0
                    nncu = []
                elif event.key == 1073741903:
                    if curn < len(activt[0].perpar[activt[1]]):
                        curn = curn + 1
                        nncu = ['l', time.time()]
                elif event.key == 1073741904:
                    if activt[0].perpar[activt[1]][curn-2:curn] != ': ':
                        curn = curn - 1
                        nncu = ['r', time.time()]
                else:
                    if event.unicode in '''QWERTYUIOPASDFGHJKLZXCVBNM qwertyuiopasdfghjklzxcvbnmйцукенгшщзхъфывапролджэячсмитьбю1234567890!@#$%^&*()_+-='"№;%:?*/\,.[]{}''':
                        if event.unicode != '':
                            activt[0].perpar[activt[1]] = activt[0].perpar[activt[1]][:curn] + event.unicode + activt[0].perpar[activt[1]][curn:]
                            if curn < len(activt[0].perpar[activt[1]]):
                                curn = curn + 1




        if (blo.check_bl() and event.type == pg.MOUSEBUTTONDOWN) or (blact and blo.check_bl()):         #       Блокнот
            blact = True
            if event.type == pg.KEYDOWN:
                if event.key == 1073741903:
                    if cur < len(blot):
                        cur += 1
                        curf = ['r', time.time()]
                elif event.key == 1073741904:
                    if cur > 0:
                        cur -= 1
                        curf = ['l', time.time()]
                elif event.key == 127:
                    blot = blot[:cur] + blot[cur+1:]
                elif event.key == 8 and blot != '':
                    blof = [time.time()]
                    if cur > 0 and blot != '':
                        cur -= 1
                        blot = blot[:cur] + blot[cur+1:]
                elif event.key == 13:
                    blot = blot[:cur] + '\n' + blot[cur:]
                    cur += 1
                elif event.key == 9:
                    blot = blot[:cur] + '    ' + blot[cur:]
                    cur += 4
                elif event.key == 1073741905:
                    if len(blot.split('\n')) > 1:
                        cur += blot[cur:].find('\n') + len(blot[:cur].split('\n')[-1]) + 1
                        curf = ['d', time.time()]
                        if cur > len(blot):
                            cur = len(blot)


                elif event.key == 1073741906:
                    if len(blot.split('\n')) > 1:
                        p = len(blot[:cur].split('\n')[-1]) - (len(blot[:cur]) - cur)
                        cur -= len(blot[:cur].split('\n')[-1]) + 1
                        if len(blot[:cur].split('\n')[-1]) - p > 0:
                            cur -= len(blot[:cur].split('\n')[-1]) - p
                        curf = ['u', time.time()]


                elif event.key == 1073742049 or event.key == 1073742050 or event.key == 1073741881 or event.key == 1073742053:
                    pass
                elif event.key == 1073742048:
                    ctrl = True
                elif event.key == 118:
                    if ctrl:
                        text = pg.scrap.get(pg.SCRAP_TEXT)
                        if text:
                            blot = blot[:cur] + str(text.decode('UTF-8'))[:-1] +blot[cur:]
                            cur += len(str(text.decode('UTF-8'))[:-1])
                    else:
                        blot = blot[:cur] + event.unicode + blot[cur:]
                        cur += 1
                else:
                    if event.unicode in '''QWERTYUIOPASDFGHJKLZXCVBNM qwertyuiopasdfghjklzxcvbnmйцукенгшщзхъфывапролджэячсмитьбюЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ1234567890!@#$%^&*()_+-='"№;%:?*/\,.[]{}''':
                        blot = blot[:cur] + event.unicode + blot[cur:]
                        cur += 1
                blot.replace('\x08', '')


            if event.type == pg.MOUSEWHEEL:
                sm += event.y*2
        else:
            blact = False




        if fls and event.type == pg.KEYDOWN:

            if event.key == 13:
                fls = False
                curs = 0


            elif event.key == 1073741903:
                    if curs < len(ts):
                        curs += 1
                        cursf = ['r', time.time()]
                        
            elif event.key == 1073741904:
                    if curs > 0:
                        curs -= 1
                        cursf = ['l', time.time()]

            elif event.key == 8 and ts != '':
                    cursf = [time.time()]
                    if curs > 0 and ts != '':
                        curs -= 1
                        ts = ts[:curs] + ts[curs+1:]


            elif event.key == 1073742049 or event.key == 1073742050 or event.key == 1073741881 or event.key == 1073742053 or event.key == 1073742048:
                    pass

            else:
                if event.unicode in '''QWERTYUIOPASDFGHJKLZXCVBNM qwertyuiopasdfghjklzxcvbnmЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮйцукенгшщзхъфывапролджэячсмитьбюіІїЇєЄ1234567890!@#$%^&*()_+-='"№;%:?*/\,.[]{}''':
                    ts = ts[:curs] + event.unicode + ts[curs:]
                    curs += 1
                ts.replace('\x08', '')





        if p1[0] < lefgr and p1[1] < ws[1] - sbl:
            if event.type == pg.MOUSEBUTTONDOWN:
                if scb.collidepoint(p1):
                    from PIL import Image
                    import pyautogui
                    myScreenshot = pyautogui.screenshot()
                    myScreenshot.save('Screenshot.png')
                    image = Image.open('Screenshot.png')
                    cropped = image.crop((poso[0], poso[1], poso[0]+lefgr, poso[1]+ws[1]-sbl-3))
                    cropped.save('Screenshot.png')



        if fls:            
            if event.type == pg.MOUSEBUTTONDOWN:        #       OK или NOT OK
                te = pg.Rect(lefgr//2-250, (ws[1]-sbl)//2-75, 500, 110)
                noto = pg.Rect(te.bottomright[0]-160, te.bottomright[1]-35, 70, 30)
                ok = pg.Rect(te.bottomright[0]-70, te.bottomright[1]-35, 35, 30)
                if noto.collidepoint(p1):
                    fls = False
                    ts = ''
                    curs = 0

                if ok.collidepoint(p1):
                    fls = False

        '''     


        if p1[0] < lefgr:                                                               #-|
            if event.type == pg.MOUSEWHEEL:                                             #  --    Начало масштабированияs
                massh = massh + event.y                                                 # /
                fon = pg.font.Font('UbuntuMono-Regular.ttf', 15+massh)                  #-|

        

        '''




        if skm != False:
            for i in rab:
                i[0].move(i[0].a.topleft[0] + p1[0] - skm[0], i[0].a.topleft[1] + p1[1] - skm[1])
            skm = p1


        if event.type == pg.MOUSEBUTTONUP:
            fl3 = False
            fl5 = False
            iz = False
            vr = False
            skm = False
            raz = False
            flrm = False
        if event.type == pg.KEYUP:
            dl = False
            blof = []
            curf = []
            cursf = []
            smf = []
            nncu = []
            if event.key == 1073742048:
                ctrl = False

    

    if fl3:
        fl3[0].move(p1[0]+smx, p1[1]+smy)
        
    
    if raz:                         #       Изменение размера нодов
        chosed = False
        fl4 = False
        fl3 = False
        if raz[2] == 'l':
            if rab[raz[0]][0].a.size[0] < 95:
                if p1[0] - raz[1][0] < 0:
                    rab[raz[0]][0].a = rab[raz[0]][0].a.move(p1[0] - raz[1][0], 0)
                    rab[raz[0]][0].a.size = (rab[raz[0]][0].a.size[0] + raz[1][0] - p1[0], rab[raz[0]][0].a.size[1])
                    raz[1] = p1
                else:
                    pg.mouse.set_pos([raz[1][0], raz[1][1]])
            else:
                rab[raz[0]][0].a = rab[raz[0]][0].a.move(p1[0] - raz[1][0], 0)
                rab[raz[0]][0].a.size = (rab[raz[0]][0].a.size[0] + raz[1][0] - p1[0], rab[raz[0]][0].a.size[1])
                raz[1] = p1
                
        else:
            if rab[raz[0]][0].a.size[0] < 95:
                if p1[0] - raz[1][0] > 0:
                    rab[raz[0]][0].a.size = (rab[raz[0]][0].a.size[0] - raz[1][0] + p1[0], rab[raz[0]][0].a.size[1])
                    raz[1] = p1
                else:
                    pg.mouse.set_pos([raz[1][0], raz[1][1]])
            else:
                rab[raz[0]][0].a.size = (rab[raz[0]][0].a.size[0] - raz[1][0] + p1[0], rab[raz[0]][0].a.size[1])
                raz[1] = p1


    for i in rab:
        '''print(has)
        if i[0].ist == 'Conv2D layer':
            print(i[0].persv)'''
        if i[0].ist == 'Model':
            mo = i[0].perpar
        i[0].create(i[1][0], i[1][1])
        if activt:
            if activt[0] == i[0]:
                pg.draw.rect(sc2, (255, 0, 0), (activt[0].params[activt[1]][1].topleft[0] + 8*curn + 5, activt[0].params[activt[1]][1].topleft[1]+2, 2, activt[0].params[activt[1]][1].size[1]-4), border_radius=7)
                pg.draw.rect(sc2, (0,0,0), (activt[0].params[activt[1]][1].topleft, activt[0].params[activt[1]][1].size), border_radius=7, width = 1)

        for j in i[0].persv:
            pg.draw.aaline(sc2, (0,20,0), [i[0].circr[j[0]].center[0]+4, i[0].circr[j[0]].center[1]],
                                              [j[1].circl[j[2]].center[0]-4, j[1].circl[j[2]].center[1]])

    if iz != False:
        pg.draw.aaline(sc2, (0,20,0), [iz.center[0]+4, iz.center[1]], p1)

    if flrm:
        sbl = ws[1] - p1[1]
        if sbl < 100:
            pg.mouse.set_pos([p1[0], ws[1]-100])

    if fl2:
        fl2.move(p1[0]+smx, p1[1]+smy)
        fl2.create(fl2.ncrl, fl2.ncrr)

    if chosed:
        pg.draw.rect(sc2, (150,150,150), (chosed[0].a.topleft[0]-1, chosed[0].a.topleft[1]-1, chosed[0].a.size[0]+1, chosed[0].a.size[1]+1), border_radius = 7, width=1)
        fl4 = False

    if len(nncu) > 0:
        if time.time() - nncu[1] > 0.5:
            if nncu[0] == 'del' and activt[0].perpar[activt[1]][-2:] != ': ' and len(activt[0].perpar[activt[1]]) != 1 and len(activt[0].perpar[activt[1]]) != 0:
                curn -= 1
                activt[0].perpar[activt[1]] = activt[0].perpar[activt[1]][:curn] + activt[0].perpar[activt[1]][curn+1:]
                time.sleep(0.05)
            elif nncu[0] == 'l' and curn < len(activt[0].perpar[activt[1]]):
                curn += 1
                time.sleep(0.05)
            elif nncu[0] == 'r' and activt[0].perpar[activt[1]][curn-2:curn] != ': ':
                curn -= 1
                time.sleep(0.05)


    
    

    if blof != [] and time.time()-blof[0] > 0.5 and cur > 0 and blot != '':
        cur -= 1
        blot = blot[:cur] + blot[cur+1:]
        time.sleep(0.03)

    if smf != [] and time.time()-smf[1] > 0.5:
        if smf[0] == 'u':
            sm += 1
        else:
            sm -= 1
        time.sleep(0.03)



    if len(curf) > 0 and curf[0] == 'l' and cur > 0 and time.time()-curf[1] > 0.5:
        cur -= 1
        time.sleep(0.03)
    if len(curf) > 0 and curf[0] == 'r' and cur < len(blot) and time.time()-curf[1] > 0.5:
        cur += 1
        time.sleep(0.03)
    if len(curf) > 0 and curf[0] == 'd' and cur < len(blot) and time.time()-curf[1] > 0.5:
        cur += blot[cur:].find('\n') + len(blot[:cur].split('\n')[-1]) + 1
        if cur > len(blot):
            cur = len(blot)
        time.sleep(0.03)
    if len(curf) > 0 and curf[0] == 'u' and cur < len(blot) and time.time()-curf[1] > 0.5:
        if len(blot.split('\n')) > 1:
            p = len(blot[:cur].split('\n')[-1]) - (len(blot[:cur]) - cur)
            cur -= len(blot[:cur].split('\n')[-1]) + 1
            if len(blot[:cur].split('\n')[-1]) - p > 0:
                cur -= len(blot[:cur].split('\n')[-1]) - p
            if cur < 0:
                cur = 0
        time.sleep(0.03)



    if len(cursf) == 1:
        if time.time()-cursf[0] > 0.5 and curs > 0 and ts != '':
            curs -= 1
            ts = ts[:curs] + ts[curs+1:]
            time.sleep(0.03)
    else:
        if len(cursf) > 0 and cursf[0] == 'l' and curs > 0 and time.time()-cursf[1] > 0.5:
            curs -= 1
            time.sleep(0.03)
        if len(cursf) > 0 and cursf[0] == 'r' and curs < len(blot) and time.time()-cursf[1] > 0.5:
            curs += 1
            time.sleep(0.03)



    if flzakr:
        lefgr = ws[0]
    else:
        lefgr = ws[0] - 250


    if fls:
        te = pg.Rect(lefgr//2-250, (ws[1]-sbl)//2-75, 500, 110)
        pg.draw.rect(sc2, (119, 127, 193), (te.topleft[0], te.topleft[1], te.size[0], te.size[1]), border_radius = 7)
        pg.draw.rect(sc2, (175, 198, 236), (te.topleft[0]+25, te.topleft[1]+25, 450, 40), border_radius = 7)
        pg.draw.rect(sc2, (255, 0, 0), (te.topleft[0]+30+curs*18+2, te.topleft[1]+28, 2, 35), border_radius = 7)



        noto = pg.Rect(te.bottomright[0]-160, te.bottomright[1]-35, 70, 30)
        pg.draw.rect(sc2, (0,128,128), noto, border_radius = 7)
        p = fon.render('Not OK', True, (255, 255, 255))
        sc2.blit(p, (te.bottomright[0]-149, te.bottomright[1]-28))



        ok = pg.Rect(te.bottomright[0]-70, te.bottomright[1]-35, 35, 30)
        pg.draw.rect(sc2, (0,128,128), ok, border_radius = 7)
        p = fon.render('Ok', True, (255, 255, 255))
        sc2.blit(p, (te.bottomright[0]-60, te.bottomright[1]-28))


        p = fons.render(ts, True, (21, 27, 44))
        sc2.blit(p, (te.topleft[0]+30, te.topleft[1]+28))





    pg.draw.aaline(sc2, (255,255,255), (0, ws[1]-sbl), (ws[0], ws[1]-sbl))
    blo.draw(sm)

    fk = pg.Rect((ws[0]-250)//2-30, 0, 60, 30)
    pg.draw.rect(sc2, (66,152,79), fk, border_bottom_left_radius = 7, border_bottom_right_radius = 7)
    p = fon.render('RUN', True, (255, 255, 255))
    sc2.blit(p, (fk.topleft[0]+20, fk.topleft[1]+7))



    scb = pg.Rect(0, 10, 100, 30)
    pg.draw.rect(sc2, (45,150,150), scb, border_bottom_right_radius = 7, border_top_right_radius = 7)
    p = fon.render('Screenshot', True, (255, 255, 255))
    sc2.blit(p, (scb.topleft[0]+10, scb.topleft[1]+7))


    razm = pg.Rect(0, ws[1]-sbl-3, ws[0], 6)
    pg.draw.rect(sc2, (0,152,0), razm)
    pg.draw.rect(sc, (0,152,0), razm)


    zap = pg.Rect(0, 50, 60, 30)
    pg.draw.rect(sc2, (45,150,150), zap, border_bottom_right_radius = 7, border_top_right_radius = 7)
    p = fon.render('Save', True, (255, 255, 255))
    sc2.blit(p, (zap.topleft[0]+12, zap.topleft[1]+7))


    op = pg.Rect(0, 90, 60, 30)
    pg.draw.rect(sc2, (45,150,150), op, border_bottom_right_radius = 7, border_top_right_radius = 7)
    p = fon.render('Open', True, (255, 255, 255))
    sc2.blit(p, (op.topleft[0]+12, op.topleft[1]+7))

    if flzakr == False:
        zakr = pg.Rect(lefgr, (ws[1]-sbl)//2-10, 10, 20)
        pg.draw.rect(sc, (0,0,100), zakr, border_top_right_radius = 3, border_bottom_right_radius = 3)
        pg.draw.rect(sc2, (0,0,100), zakr, border_top_right_radius = 3, border_bottom_right_radius = 3)
    else:
        zakr = pg.Rect(lefgr-10, (ws[1]-sbl)//2-10, 10, 20)
        pg.draw.rect(sc2, (0,0,100), zakr, border_top_left_radius = 3, border_bottom_left_radius = 3)
        pg.draw.rect(sc, (0,0,100), zakr, border_top_left_radius = 3, border_bottom_left_radius = 3)

    
    clock.tick(60)
    window.blit(sc, (0,0))
    window.blit(sc2, (0,0))
    window.blit(bl, (0, ws[1]-sbl))
    sc = pg.Surface((ws[0], ws[1]-sbl))
    sc2 = pg.Surface((lefgr, ws[1]-sbl))
    bl = pg.Surface((ws[0], sbl))
    pg.display.update()

    sc.fill((225,238,240))
    sc2.fill((171,205,239))
    
    '''sc.fill((30,30,30))
    sc2.fill((30,30,30))'''
    bl.fill((0, 0, 20))
    #clock.tick(80)


