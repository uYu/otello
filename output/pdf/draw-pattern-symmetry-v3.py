from pathlib import Path
import hashlib,html,json,math,struct
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
model=ROOT/'models/pattern32-symmetry-v1.37.otpat';m=json.loads(model.with_suffix('.json').read_text())
assert hashlib.sha256(model.read_bytes()).hexdigest()==m['sha256']
assert struct.unpack_from('<8s7I',model.read_bytes())==(b'OTPAT4\0\0',28503,32,32,97,46,256,1)
pdfmetrics.registerFont(TTFont('CN','/System/Library/Fonts/STHeiti Medium.ttc',subfontIndex=0))
W,H=1800,1050
c=canvas.Canvas(str(OUT/'pattern-symmetry-v1.37.pdf'),pagesize=(W,H));c.setTitle('D4 symmetry-tied Pattern policy-value network');c.setAuthor('Otello')
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>']
navy='#182C45';muted='#617083';edge='#CCD7E3';teal='#13857E';blue='#416AB5';purple='#7954B7';orange='#E6A455';white='#FFFFFF'

def color(x):return HexColor(x)
def txt(x,y,s,size=22,font='CN',fill=navy,anchor='middle'):
 c.setFillColor(color(fill));c.setFont(font,size)
 {'start':c.drawString,'middle':c.drawCentredString,'end':c.drawRightString}[anchor](x,y,s)
 family='Heiti SC, PingFang SC, sans-serif' if font=='CN' else 'Arial, Helvetica, sans-serif' if 'Helvetica' in font else 'Times New Roman, serif'
 svg.append(f'<text x="{x}" y="{H-y}" text-anchor="{anchor}" font-family="{family}" font-size="{size}" fill="{fill}"'+(' font-weight="bold"' if 'Bold' in font else '')+(' font-style="italic"' if 'Italic' in font else '')+'>'+html.escape(s)+'</text>')
def poly(pts,fill,stroke=None,lw=1):
 c.setFillColor(color(fill));c.setStrokeColor(color(stroke or fill));c.setLineWidth(lw)
 p=c.beginPath();p.moveTo(*pts[0])
 for xy in pts[1:]:p.lineTo(*xy)
 p.close();c.drawPath(p,fill=1,stroke=bool(stroke))
 svg.append('<polygon points="'+' '.join(f'{x},{H-y}' for x,y in pts)+f'" fill="{fill}" stroke="{stroke or fill}" stroke-width="{lw}"/>')
def line(pts,fill=muted,lw=2,arrow=False,dash=False):
 c.setStrokeColor(color(fill));c.setLineWidth(lw);c.setDash([5,4] if dash else [])
 p=c.beginPath();p.moveTo(*pts[0])
 for xy in pts[1:]:p.lineTo(*xy)
 c.drawPath(p);c.setDash([])
 svg.append('<polyline points="'+' '.join(f'{x},{H-y}' for x,y in pts)+f'" fill="none" stroke="{fill}" stroke-width="{lw}"'+(' stroke-dasharray="5 4"' if dash else '')+'/>')
 if arrow:
  (a,b),(x,y)=pts[-2:];ang=math.atan2(y-b,x-a);r=10;t=4.8
  poly([(x,y),(x-r*math.cos(ang)+t*math.sin(ang),y-r*math.sin(ang)-t*math.cos(ang)),(x-r*math.cos(ang)-t*math.sin(ang),y-r*math.sin(ang)+t*math.cos(ang))],fill)
def box(x,y,w,h,fill=white,stroke=edge,r=10,lw=1.5):
 c.setFillColor(color(fill));c.setStrokeColor(color(stroke));c.setLineWidth(lw);c.roundRect(x,y,w,h,r,fill=1,stroke=1)
 svg.append(f'<rect x="{x}" y="{H-y-h}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{lw}"/>')
def circ(x,y,r,fill,stroke=None,lw=1.5):
 c.setFillColor(color(fill));c.setStrokeColor(color(stroke or fill));c.setLineWidth(lw);c.circle(x,y,r,fill=1,stroke=bool(stroke))
 svg.append(f'<circle cx="{x}" cy="{H-y}" r="{r}" fill="{fill}" stroke="{stroke or fill}" stroke-width="{lw}"/>')
def panel(x,y,w,h,n,title,tint):
 box(x,y,w,h,white,edge,r=16)
 circ(x+35,y+h-38,18,tint);txt(x+35,y+h-45,n,20,'Helvetica-Bold',white)
 txt(x+70,y+h-47,title,27,fill=tint,anchor='start')
def grid(x,y,n=8,step=12,discs=None,heat=False):
 for yy in range(n):
  for xx in range(n):
   fill='#EFF3F7'
   if heat and (yy,xx) in [(2,3),(3,2),(4,5),(5,4)]:fill=purple
   box(x+xx*step,y+yy*step,step-.5,step-.5,fill,'#CBD5DF',r=0,lw=.5)
 if discs is not None:
  for yy,row in enumerate(discs):
   for xx,v in enumerate(row):
    if v:circ(x+(xx+.5)*step,y+(n-yy-.5)*step,step*.34,navy if v==1 else white,navy,1)
def tensor(x,y,w,h,rows,cols,tint,light,depth=12,highlight=False):
 poly([(x,y+h),(x+depth,y+h+depth),(x+w+depth,y+h+depth),(x+w,y+h)],light,tint,1)
 poly([(x+w,y),(x+w+depth,y+depth),(x+w+depth,y+h+depth),(x+w,y+h)],light,tint,1)
 box(x,y,w,h,white,tint,r=0,lw=1.4)
 for rr in range(rows):
  for cc in range(cols):
   f=light if (rr+cc)%3 else white
   if highlight and rr==3:f='#F4C572'
   box(x+cc*w/cols,y+rr*h/rows,w/cols-.6,h/rows-.6,f,white,r=0,lw=.5)

def strip(x,y,values,step=12):
 for i,v in enumerate(values):box(x+i*step,y,step-1,20,teal if v==1 else navy if v==2 else '#E3EDEB',white,r=2,lw=.5)

# Hero: one visual thesis, then three numbered functional blocks.
txt(60,977,'对称共享 Pattern 网络',44,fill=navy,anchor='start')
txt(60,942,'D4-TIED PATTERN POLICY-VALUE NETWORK  /  v1.37 candidate',22,'Helvetica',muted,'start')
box(1320,940,420,78,'#E9F5F2','#C6E2DC',r=14)
txt(1340,986,'0.917M',32,'Helvetica-Bold',teal,'start');txt(1518,986,'参数',24,fill=teal,anchor='start')
txt(1340,956,'1.788M → 0.917M',20,fill=muted,anchor='start');txt(1718,956,'减少 48.7%',22,fill=teal,anchor='end')
panel(60,485,810,425,'1','对称编码与共享查表',teal)
panel(930,485,810,425,'2','全局聚合与价值判断',blue)
panel(930,80,810,345,'3','条件策略头',purple)
box(60,80,810,345,'#F5FAF9','#CEE3DF',r=16)
txt(94,373,'真正共享的是：同一类棋形的对称权重',27,fill=teal,anchor='start')

# (1) Board -> patterns -> canonical folding -> shared table.
board=[[0]*8 for _ in range(8)];board[3][3]=board[4][4]=1;board[3][4]=board[4][3]=2
grid(90,637,8,13,board)
txt(142,606,'行棋方视角',21);txt(142,577,'8 × 8 棋盘',19,fill=muted)
line([(205,690),(252,690)],arrow=True)
# A stack of line, diagonal and corner patterns.
for k in range(3):box(268+k*6,649+k*9,92,84,white,'#A5BBB7',r=4,lw=1)
for k in range(7):circ(283+10*k,710,3.5,teal)
for k in range(6):circ(284+10*k,692-5*k,3.5,blue)
for yy in range(3):
 for xx in range(3):box(322+8*xx,655+8*yy,6,6,'#C5DCD7',teal,r=0,lw=.5)
txt(318,606,'38 个棋形',22);txt(318,576,'16 / 18 / 4',16,'Helvetica',muted);txt(318,552,'行列 / 对角 / 角部',15,fill=muted)
line([(375,690),(412,690)],arrow=True)
seq=[0,1,2,0,0,1,2,0];strip(426,717,seq);strip(426,652,seq[::-1])
txt(474,746,'x',20,'Times-Italic',teal);txt(474,681,'reverse(x)',17,'Times-Italic',teal)
line([(526,727),(550,727),(550,690)],teal,2)
line([(526,662),(550,662),(550,690)],teal,2)
line([(550,690),(625,690)],teal,2.5,True)
circ(550,690,4,teal)
txt(495,606,'D4 规范索引',22,fill=teal)
txt(495,578,'对称输入 → 同一行',18,fill=muted)
tensor(645,630,160,125,8,8,teal,'#D3EBE5',16,True)
txt(727,602,'28,503 × 32',23,'Helvetica-Bold',teal)
txt(727,573,'共享 INT16 表',20,fill=muted)
box(255,505,550,40,'#E9F5F2','#E9F5F2',r=7)
txt(530,517,'55,728 行合并为 28,503 行；输出 38 个 32 维向量',20,fill=teal)
# E flows to (2), keeping main feature route on one horizontal line.
line([(821,690),(955,690)],teal,3,True)

# (2) Tensor, three region pools, concatenate scalar, 97->32 shared trunk.
tensor(960,645,95,105,8,5,teal,'#DAEDEA',10)
txt(1007,777,'棋形特征 E',22,fill=teal)
txt(1007,607,'38 × 32',22,'Helvetica-Bold',teal)
line([(1065,690),(1118,690)],arrow=True)
for x,col,label,num in [(1125,teal,'边',4),(1166,blue,'内',30),(1207,purple,'角',4)]:
 box(x,647,25,90,col,col,r=4);txt(x+12,617,label,19,fill=col);txt(x+12,592,str(num),16,'Helvetica',muted)
txt(1180,777,'三组均值池化',23,fill=blue)
txt(1180,567,'3 × 32 = 96',19,'Helvetica',muted)
line([(1232,690),(1262,690)],arrow=True)
circ(1280,690,16,white,blue);txt(1280,684,'||',20,'Helvetica-Bold',blue)
txt(1280,723,'拼接',18,fill=muted)
line([(1296,690),(1320,690)],arrow=True)
box(1201,514,158,43,'#F2F5FA',edge,r=7);txt(1280,527,'空格数 / 60',20,fill=blue)
line([(1280,557),(1280,674)],blue,2,True);txt(1294,596,'+1',17,'Helvetica',muted,'start')
# Small actual neural topology rather than another paragraph box.
box(1322,624,125,133,'#F0F4FB','#C4D3EB',r=9)
for left in [648,673,698,723]:
 for right in [649,679,709,739]:line([(1344,left),(1424,right)],'#CCDAEF',.7)
for yy in [648,673,698,723]:circ(1344,yy,6,white,blue,1.5)
for yy in [649,679,709,739]:circ(1424,yy,7,blue,blue)
txt(1385,777,'97 → 32',25,'Helvetica-Bold',blue)
txt(1385,597,'clipped ReLU',18,'Helvetica',muted)
txt(1385,570,'h ∈ R³²',20,fill=blue)
# Two value heads, short and explicit.
line([(1447,690),(1474,690)],blue,2.3)
line([(1474,623),(1474,750)],blue,2.3)
for yy in [750,623]:line([(1474,yy),(1510,yy)],blue,2.3,True)
box(1510,711,190,80,'#EAF0FA','#B9CBE6',r=10)
txt(1605,760,'胜负价值 v(s)',25,fill=blue);txt(1605,731,'32 → 1 → tanh',20,fill=blue)
box(1510,584,190,80,'#F4F6F9',edge,r=10)
txt(1605,634,'辅助 WDL',25,fill=muted);txt(1605,604,'32 → 3 → softmax',18,fill=muted)
txt(1605,550,'价值用于搜索',16,fill=muted);txt(1605,525,'WDL 仅作训练辅助',16,fill=muted)
# Model invariant excerpt at bottom-left: two corners feed a single embedding row.
corner=[[1,0,2],[2,1,0],[0,0,1]];transposed=[list(x) for x in zip(*corner)]
grid(100,225,3,25,corner);grid(270,225,3,25,transposed)
txt(137,199,'角部 X',20);txt(307,199,'转置 X',20)
# Each input is shown feeding the same row; separate curved-level routes.
line([(180,263),(210,263),(210,323),(402,323),(402,263)],teal,2)
line([(350,263),(402,263)],teal,2)
line([(402,263),(450,263)],teal,2.5,True);circ(402,263,4,teal)
txt(573,319,'同一个规范索引',23,fill=teal)
for j in range(24):box(452+13*j,246,12,34,['#71B5A7','#A5D2C7','#D1E9E2','#2B9689'][j%4],white,r=1,lw=.5)
txt(608,214,'同一个 32 维 embedding',22,fill=teal)
txt(465,144,'旋转 / 镜像后：价值不变，策略随棋盘变换',26,fill=teal)
txt(465,110,'v(Ts) = v(s)       π(Ta | Ts) = π(a | s)',23,fill=muted)

# (3) Algebraically equivalent presentation of the current conditional policy.
line([(960,670),(943,670),(943,247),(985,247)],teal,2.8,True)
txt(960,453,'局部 E',21,fill=teal,anchor='start')
line([(1460,690),(1460,440),(1360,440),(1360,305)],blue,2.8,True);circ(1460,690,3,blue)
txt(1480,453,'全局 h',21,fill=blue,anchor='start')
box(985,200,165,95,'#EAF5F2','#BDDAD2',r=10)
txt(1067,259,'格点特征',25,fill=teal);txt(1067,222,'f_a [32]',23,'Times-Italic',teal)
txt(1067,176,'关联棋形取均值',18,fill=muted)
line([(1150,247),(1225,247)],teal,2.6,True)
box(1225,190,270,115,'#F2EDF9','#D7C8EA',r=10)
txt(1360,275,'共享落点打分器',25,fill=purple)
txt(1360,239,'<f_a, g(h)> / sqrt(32)',21,'Times-Italic',purple)
txt(1360,210,'+ b(h)[bucket(a)]',21,'Times-Italic',purple)
txt(1376,331,'h 生成门控 g 与位置偏置 b',18,fill=blue,anchor='start')
line([(1495,247),(1535,247)],purple,2.6,True)
box(1535,200,175,95,'#EAF0FA','#B9CBE6',r=10)
txt(1622,259,'合法 softmax',21,fill=blue);txt(1622,222,'π(a | s)',24,fill=blue)
txt(1622,176,'PUCT 先验',20,fill=blue)
txt(1335,127,'f_a：覆盖落点 a 的棋形向量均值；每个落点复用同一套打分参数',18,fill=muted)
txt(1335,100,'当前策略头 1,386 参数 · 本图为代数等价表达，未改变候选权重',18,fill=muted)

line([(60,55),(1740,55)],edge,1)
txt(60,28,'图 1  对称共享只改变查表参数绑定；38 个棋形、三组池化、97→32 主干与价值/策略输出保持原结构。',20,fill=muted,anchor='start')
c.showPage();c.save();svg.append('</svg>');(OUT/'pattern-symmetry-v1.37.svg').write_text('\n'.join(svg)+'\n')
print('Created redesigned vector architecture')
