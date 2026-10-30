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
W,H=1600,680
c=canvas.Canvas(str(OUT/'policy-head-direct-mlp8-proposal.pdf'),pagesize=(W,H));c.setTitle('Direct shared policy head - untrained prototype');c.setAuthor('Otello')
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


txt(55,611,'直连共享策略头',43,fill=navy,anchor='start')
txt(55,571,'结构候选 / 原型已实现，尚未训练及对战',23,fill=muted,anchor='start')
box(1160,562,370,84,'#F0EAF8','#D6C8E9',r=13)
txt(1345,611,'固定权重 · 单一路径',29,fill=purple)
txt(1345,578,'local + global + position → score',18,'Helvetica',muted)
# Straight feature route.
tensor(65,291,105,105,7,5,teal,'#D4EAE4',12)
txt(118,430,'棋形查表 E',26,fill=teal);txt(118,252,'38 × 32',23,'Helvetica',teal)
line([(182,346),(230,346)],teal,2.8,True)
box(230,290,180,112,'#EDF6F3','#C5DFD7',r=11)
txt(320,365,'落点特征 f_a',26,fill=teal);txt(320,326,'64 × 32',25,'Helvetica-Bold',teal)
txt(320,254,'覆盖该格点的棋形取均值',19,fill=muted)
line([(410,346),(485,346)],teal,2.8,True)
box(485,290,185,112,'#F2F4FA','#CDD7EA',r=11)
txt(577,365,'直接拼接',27,fill=blue);txt(577,326,'[ f_a, h, pos ]',23,'Times-Italic',blue)
txt(577,300,'32 + 32 + 10 = 74',16,'Helvetica',muted)
# Context is an input, never a generator of another layer's weights.
box(485,467,185,72,'#EDF2FA','#C6D4E9',r=10)
txt(577,510,'全局 h[32]',25,fill=blue);txt(577,482,'来自共享主干',19,fill=muted)
line([(577,467),(577,402)],blue,2.4,True)
box(485,117,185,72,'#FAF2E8','#E7D5BB',r=10)
txt(577,160,'位置编码 [10]',23,fill='#9C6A32');txt(577,132,'D4 不变的 one-hot',18,fill=muted)
line([(577,189),(577,290)],'#B9894D',2.4,True)
line([(670,346),(750,346)],blue,2.8,True)
# All candidate moves use the exact same fixed MLP parameters.
box(750,267,310,156,'#F3EDF9','#D3C3E7',r=13)
txt(905,389,'共享 MLP',30,fill=purple)
for yy in [308,330,352]:
 for zz in [309,331,353]:line([(810,yy),(907,zz)],'#D8C9EC',1)
for yy in [308,330,352]:circ(810,yy,5.5,white,purple,1.6)
for yy in [309,331,353]:circ(907,yy,7,purple)
for yy in [309,331,353]:line([(914,yy),(993,331)],'#B99BD9',1.3)
circ(1001,331,7,purple)
txt(905,282,'74 → 8 → 1',24,'Helvetica-Bold',purple)
txt(905,230,'中间层：clipped ReLU',20,fill=muted)
line([(1060,346),(1110,346)],purple,2.8,True)
box(1110,290,150,112,'#F4EFF9','#D6C7E8',r=11)
txt(1185,365,'64 格分数',25,fill=purple);txt(1185,327,'logits',25,'Times-Italic',purple)
line([(1260,346),(1310,346)],purple,2.8,True)
box(1310,290,220,112,'#ECF1FA','#C1D0E8',r=11)
txt(1420,366,'合法掩码 + softmax',23,fill=blue);txt(1420,325,'π(a | s)',28,fill=blue)
txt(1420,252,'PUCT 搜索先验',22,fill=blue)
line([(55,83),(1530,83)],edge,1)
txt(55,52,'每个落点直接使用局部、全局与位置特征，通过同一个固定 MLP 评分。没有门控乘法，也没有独立偏置分支。',21,fill=muted,anchor='start')
txt(55,21,'设计保持 D4 等变；这是需要重训验证的新头，当前 v1.37 模型与正在进行的对战保持冻结。',20,fill=muted,anchor='start')
c.showPage();c.save();svg.append('</svg>');(OUT/'policy-head-direct-mlp8-proposal.svg').write_text('\n'.join(svg)+'\n')
