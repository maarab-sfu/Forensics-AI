# Copyright 2023 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


from dataclasses import dataclass
from copy import deepcopy
import torch

import random
import uuid

import numpy as np 
from typing import List, Tuple
from copy import deepcopy


class BCH(object):

   @dataclass
   class params:
      m: int
      t: int
      poly: int

   @dataclass
   class polynomial:
      deg: int



   ### GALOIS OPERATIONS

   def g_inv(self,a):
      return self.ECCstate.exponents[self.ECCstate.n - self.ECCstate.logarithms[a]]    

   def g_sqrt(self, a):
      if a:
         return self.ECCstate.exponents[self.mod(2*self.ECCstate.logarithms[a])]
      else:
         return 0

   def mod(self, v):
       if v<self.ECCstate.n:
         return v
       else:
         return v-self.ECCstate.n

   def g_mul(self, a,b):

      if (a>0 and b>0):
        res=self.mod(self.ECCstate.logarithms[a]+self.ECCstate.logarithms[b])
        return (self.ECCstate.exponents[res])
      else:
        return 0

   def g_div(self,a,b):
      if a:
        return self.ECCstate.exponents[self.mod(self.ECCstate.logarithms[a]+self.ECCstate.n-self.ECCstate.logarithms[b])]
      else:
        return 0

   def modn(self, v):
      n=self.ECCstate.n
      while (v>=n):
         v -= n
         v = (v & n) + (v >> self.ECCstate.m)
      return v

   def g_log(self, x):
      return self.ECCstate.logarithms[x]

   def a_ilog(self, x):
      return self.mod(self.ECCstate.n- self.ECCstate.logarithms[x])

   def g_pow(self, i):
      return self.ECCstate.exponents[self.modn(i)]

   def deg(self, x):
      count=0
      while (x >> 1):
          x = x >> 1
          count += 1
      return count

   
   def ceilop(self, a, b):
      return  int((a + b - 1) / b)

   def load4bytes(self, data):
      w=0
      w += data[0] << 24
      w += data[1] << 16
      w += data[2] << 8
      w += data[3] << 0
      return w


   def getroots(self, k, poly):

      roots=[]
      
      if poly.deg>2: 
         k=k*8+self.ECCstate.ecc_bits

         rep=[0]*(self.ECCstate.t*2)
         d=poly.deg
         l=self.ECCstate.n-self.g_log(poly.c[poly.deg])
         for i in range(0,d):
            if poly.c[i]:
              rep[i]=self.mod(self.g_log(poly.c[i])+l)
            else:
              rep[i]=-1

         rep[poly.deg]=0
         syn0=self.g_div(poly.c[0],poly.c[poly.deg])
         for i in range(self.ECCstate.n-k+1, self.ECCstate.n+1):
             syn=syn0
             for j in range(1,poly.deg+1):
                 m=rep[j]
                 if m>=0:
                    syn = syn ^ self.g_pow(m+j*i)
             if syn==0:
                 roots.append(self.ECCstate.n-i)
                 if len(roots)==poly.deg:
                     break
         if len(roots)<poly.deg:
             # not enough roots to correct
             self.ECCstate.errloc=[]
             return -1

      if poly.deg==1:
         if (poly.c[0]):
            roots.append(self.mod(self.ECCstate.n-self.ECCstate.logarithms[poly.c[0]]+self.ECCstate.logarithms[poly.c[1]]) )

      if poly.deg==2:
         if (poly.c[0] and poly.c[1]):
            l0=self.ECCstate.logarithms[poly.c[0]]        
            l1=self.ECCstate.logarithms[poly.c[1]]        
            l2=self.ECCstate.logarithms[poly.c[2]]        

            u=self.g_pow(l0+l2+2*(self.ECCstate.n-l1))
            r=0
            v=u
            while (v):
               i=self.deg(v)
               r = r ^ self.ECCstate.elp_pre[i]
               v = v ^ pow(2,i)
            if self.g_sqrt(r)^r == u:
               roots.append(self.modn(2*self.ECCstate.n-l1-self.ECCstate.logarithms[r]+l2))
               roots.append(self.modn(2*self.ECCstate.n-l1-self.ECCstate.logarithms[r^1]+l2))


      self.ECCstate.errloc=roots  
      return len(roots)

   def get_ecc_bits(self):
      return self.ECCstate.ecc_bits


   def get_ecc_bytes(self):
      return self.ceilop(self.ECCstate.m * self.ECCstate.t, 8)


   def decode(self,data,recvecc): 
                
      calc_ecc=self.encode(data)
                
      self.ECCstate.errloc=[]
                
      ecclen=len(recvecc)
      mlen=int(ecclen/4) # how many whole words
      eccbuf=[]
      offset=0
      while (mlen>0):
         w=self.load4bytes(recvecc[offset:(offset+4)])
         eccbuf.append(w)
         offset+=4
         mlen -=1
      recvecc=recvecc[offset:]
      leftdata=len(recvecc)
      if leftdata>0: #pad it to 4
        recvecc=recvecc+bytes([0]*(4-leftdata))
        w=self.load4bytes(recvecc)
        eccbuf.append(w)

      eccwords=self.ceilop(self.ECCstate.m*self.ECCstate.t, 32)
      
      sum=0
      for i in range(0,eccwords):
         self.ECCstate.ecc_buf[i] = self.ECCstate.ecc_buf[i] ^ eccbuf[i]
         sum = sum | self.ECCstate.ecc_buf[i]
      if sum==0:
        return 0 # no bit flips
      

      s=self.ECCstate.ecc_bits
      t=self.ECCstate.t
      syn=[0]*(2*t)  
      
      m= s & 31  

      synbuf=self.ECCstate.ecc_buf
      
      if (m):
        synbuf[int(s/32)] = synbuf[int(s/32)] & ~(pow(2,32-m)-1)
         
      synptr=0
      while(s>0 or synptr==0):
          poly=synbuf[synptr]
          synptr += 1
          s-= 32
          while (poly):
             i=self.deg(poly)
             for j in range(0,(2*t),2):
               syn[j]=syn[j] ^ self.g_pow((j+1)*(i+s))
             poly = poly ^ pow(2,i)
         
         
      for i in range(0,t):
         syn[2*i+1]=self.g_sqrt(syn[i])
             

      n=self.ECCstate.n
      t=self.ECCstate.t
      pp=-1
      pd=1
      
      pelp=self.polynomial(deg=0)
      pelp.deg=0
      pelp.c= [0]*(2*t)
      pelp.c[0]=1
      
      elp=self.polynomial(deg=0)
      elp.c= [0]*(2*t)
      elp.c[0]=1
          
      d=syn[0]
             
      elp_copy=self.polynomial(deg=0)  
      for i in range(0,t):
         if (elp.deg>t):
             break
         if d:
            k=2*i-pp
            elp_copy=deepcopy(elp)
            tmp=self.g_log(d)+n-self.g_log(pd)
            for j in range(0,(pelp.deg+1)):
              if (pelp.c[j]):
                l=self.g_log(pelp.c[j])
                elp.c[j+k]=elp.c[j+k] ^ self.g_pow(tmp+l)


            tmp=pelp.deg+k
            if tmp>elp.deg:
                elp.deg=tmp
                pelp=deepcopy(elp_copy)
                pd=d
                pp=2*i
         if (i<t-1):
             d=syn[2*i+2]
             for j in range(1,(elp.deg+1)):
                 d = d ^ self.g_mul(elp.c[j],syn[2*i+2-j])
      self.ECCstate.elp=elp


      nroots = self.getroots(len(data),self.ECCstate.elp)
      datalen=len(data)
      nbits=(datalen*8)+self.ECCstate.ecc_bits

      for i in range(0,nroots):
          if self.ECCstate.errloc[i] >= nbits:
            return -1
          self.ECCstate.errloc[i]=nbits-1-self.ECCstate.errloc[i]
          self.ECCstate.errloc[i]=(self.ECCstate.errloc[i] & ~7) | (7-(self.ECCstate.errloc[i] & 7))
        

      for bitflip in self.ECCstate.errloc:
          byte= int (bitflip / 8)
          bit = pow(2,(bitflip & 7))
          if bitflip < (len(data)+len(recvecc))*8:
            if byte<len(data):
              data[byte] = data[byte] ^ bit
            else:
              recvecc[byte - len(data)] = recvecc[byte - len(data)] ^ bit


      return nroots


   def encode(self,data):

      datalen=len(data)
      l=self.ceilop(self.ECCstate.m*self.ECCstate.t, 32)-1

      ecc= [0]*self.ECCstate.ecc_bytes

      ecc_max_words=self.ceilop(31*64, 32)
      r = [0]*ecc_max_words

      tab0idx=0
      tab1idx=tab0idx+256*(l+1)
      tab2idx=tab1idx+256*(l+1)
      tab3idx=tab2idx+256*(l+1)
  
      mlen=int(datalen/4) # how many whole words
      offset=0
      while (mlen>0):
         w=self.load4bytes(data[offset:(offset+4)])
         w=w^r[0]
         p0=tab0idx+(l+1)*((w>>0) & 0xff)
         p1=tab1idx+(l+1)*((w>>8) & 0xff)
         p2=tab2idx+(l+1)*((w>>16) & 0xff)
         p3=tab3idx+(l+1)*((w>>24) & 0xff)
      
         for i in range(0,l):
           r[i]=r[i+1] ^ self.ECCstate.cyclic_tab[p0+i] ^ self.ECCstate.cyclic_tab[p1+i] ^ self.ECCstate.cyclic_tab[p2+i] ^ self.ECCstate.cyclic_tab[p3+i]

         r[l] = self.ECCstate.cyclic_tab[p0+l]^self.ECCstate.cyclic_tab[p1+l]^self.ECCstate.cyclic_tab[p2+l]^self.ECCstate.cyclic_tab[p3+l];
         mlen -=1
         offset +=4


      data=data[offset:]
      leftdata=len(data)
      
      ecc=r
      posn=0
      while (leftdata):
          tmp=data[posn]
          posn += 1
          pidx = (l+1)*(((ecc[0] >> 24)^(tmp)) & 0xff)
          for i in range(0,l):
             ecc[i]=(((ecc[i] << 8)&0xffffffff)|ecc[i+1]>>24)^(self.ECCstate.cyclic_tab[pidx])
             pidx += 1
          ecc[l]=((ecc[l] << 8)&0xffffffff)^(self.ECCstate.cyclic_tab[pidx])
          leftdata -= 1

      self.ECCstate.ecc_buf=ecc
      eccout=[]
      for e in r:
         eccout.append((e >> 24) & 0xff)
         eccout.append((e >> 16) & 0xff)
         eccout.append((e >> 8) & 0xff)
         eccout.append((e >> 0) & 0xff)

      eccout=eccout[0:self.ECCstate.ecc_bytes]

      eccbytes=(bytearray(bytes(eccout)))
      return eccbytes



   def build_cyclic(self, g):

     l=self.ceilop(self.ECCstate.m*self.ECCstate.t, 32)

     plen=self.ceilop(self.ECCstate.ecc_bits+1,32)
     ecclen=self.ceilop(self.ECCstate.ecc_bits,32)

     self.ECCstate.cyclic_tab = [0] * 4*256*l
   
     for i in range(0,256):
        for b in range(0,4):
          offset= (b*256+i)*l
          data = i << 8*b
          while (data):

            d=self.deg(data)
            data = data ^ (g[0] >> (31-d))
            for j in range(0,ecclen):
               if d<31:
                 hi=(g[j] << (d+1)) & 0xffffffff
               else:
                 hi=0
               if j+1 < plen:
                 lo= g[j+1] >> (31-d)
               else:
                 lo= 0
               self.ECCstate.cyclic_tab[j+offset] = self.ECCstate.cyclic_tab[j+offset] ^ (hi | lo)


   def __init__(self, t, poly):

      tmp = poly;
      m = 0;
      while (tmp >> 1):
         tmp =tmp >> 1 
         m +=1  
 
      self.ECCstate=self.params(m=m,t=t,poly=poly)

      self.ECCstate.n=pow(2,m)-1
      words = self.ceilop(m*t,32)
      self.ECCstate.ecc_bytes = self.ceilop(m*t,8)
      self.ECCstate.cyclic_tab=[0]*(words*1024)
      self.ECCstate.syn=[0]*(2*t)
      self.ECCstate.elp=[0]*(t+1)
      self.ECCstate.errloc=[0] * t
 

      x=1
      k=pow(2,self.deg(poly))
      if k != pow(2,self.ECCstate.m):
        return -1

      self.ECCstate.exponents=[0]*(1+self.ECCstate.n)
      self.ECCstate.logarithms=[0]*(1+self.ECCstate.n)
      self.ECCstate.elp_pre=[0]*(1+self.ECCstate.m)
   
      for i in range(0,self.ECCstate.n):
        self.ECCstate.exponents[i]=x
        self.ECCstate.logarithms[x]=i
        if i and x==1:
         return -1
        x*= 2
        if (x & k):
          x=x^poly

      self.ECCstate.logarithms[0]=0
      self.ECCstate.exponents[self.ECCstate.n]=1



      n=0
      g=self.polynomial(deg=0)   
      g.c=[0]*((m*t)+1) 
      roots=[0]*(self.ECCstate.n+1)
      genpoly=[0]*self.ceilop(m*t+1,32)

      # enum all roots
      for i in range(0,t):
         r=2*i+1
         for j in range(0,m):
           roots[r]=1
           r=self.mod(2*r)

      # build g(x)
      g.deg=0
      g.c[0]=1
      for i in range(0,self.ECCstate.n):
        if roots[i]:
          r=self.ECCstate.exponents[i]
          g.c[g.deg+1]=1
          for j in range(g.deg,0,-1):
              g.c[j]=self.g_mul(g.c[j],r)^g.c[j-1]
          g.c[0]=self.g_mul(g.c[0],r)
          g.deg += 1

      # store
      n = g.deg+1
      i = 0

      while (n>0) :

         if n>32:
            nbits=32
         else:
            nbits=n

         word=0
         for j in range (0,nbits):
           if g.c[n-1-j] :
               word = word | pow(2,31-j)
         genpoly[i]=word
         i += 1
         n -= nbits
      self.ECCstate.ecc_bits=g.deg

      self.build_cyclic(genpoly);


      sum=0
      aexp=0
      for i in range(0,m):
        for j in range(0,m):
          sum = sum ^ self.g_pow(i*pow(2,j))
        if sum: 
          aexp=self.ECCstate.exponents[i]
          break

      x=0
      precomp=[0] * 31
      remaining=m

      while (x<= self.ECCstate.n and remaining):
         y=self.g_sqrt(x)^x
         for i in range(0,2):
            r=self.g_log(y)
            if (y and (r<m) and not precomp[r]):
              self.ECCstate.elp_pre[r]=x
              precomp[r]=1
              remaining -=1
              break
            y=y^aexp
         x += 1


BCH_POLYNOMIAL = 137

class DataLayer(object):
    def __init__(self, payload_len, verbose=True, encoding_mode=0, **kw_args):

        self.bch_encoder=self.buildBCH(encoding_mode)
        self.encoding_mode=encoding_mode
        self.versionbits=4

        self.bch_decoders=dict()
        for i in range(0,7):
          self.bch_decoders[i]=self.buildBCH(i)
        self.payload_len = payload_len  # in bits

    def schemaInfo(self, version):
        if version==0:
            return 'BCH_SUPER'
        if version==1:
            return 'BCH_5'
        if version==2:
            return 'BCH_4'
        if version==3:
            return 'BCH_3'
        if version == 4:
            return 'BCH_6'
        return 'Unknown'

    def schemaCapacity(self, version):
        if version==0:
            return 40
        if version==1:
            return 61
        if version==2:
            return 68
        if version==3:
            return 75
        if version==4:
            return 52
        return 0


    def buildBCH(self, encoding_mode):
        if encoding_mode==1:
             return (BCH(5,BCH_POLYNOMIAL))
        elif encoding_mode==2:
             return (BCH(4,BCH_POLYNOMIAL))
        elif encoding_mode==3:
             return (BCH(3,BCH_POLYNOMIAL))
        elif encoding_mode==4:
             return (BCH(6,BCH_POLYNOMIAL))
        else:  # assume superwatermark/mode 0
             return(BCH(8,BCH_POLYNOMIAL))


    def raw_payload_split(self, packet):

        packet = ''.join([str(int(bit)) for bit in packet])  # bit string

        wm_version = int(packet[-(self.versionbits-2):],2) # from last 2 bits of the 4 version bits
#        print('Found watermark with encoding schema %s' % self.schemaInfo(wm_version))

        if wm_version==1:
                # 5 bitflips via 35 ecc bits, 61 bit payload
                data=packet[0:61]
                ecc=packet[61:96]
                bitflips=5
                decoder=self.bch_decoders[1]
        elif wm_version==2:
                # 4 bitflips via 28 ecc bits, 68 bit payload
                data=packet[0:68]
                ecc=packet[68:96]
                bitflips=4
                decoder=self.bch_decoders[2]
        elif wm_version==3:
                # 3 bitflips via 21 ecc bits, 75 bit payload
                data=packet[0:75]
                ecc=packet[75:96]
                bitflips=3
                decoder=self.bch_decoders[3]
        elif wm_version==4:
                # 3 bitflips via 44 ecc bits, 52 bit payload
                data=packet[0:52]
                ecc=packet[52:96]
                bitflips=6
                decoder=self.bch_decoders[4]
        elif wm_version==0:
                # 8 bitflips via 56 ecc bits, 40 bit payload
                data=packet[0:40]
                ecc=packet[40:96]
                bitflips=8
                decoder=self.bch_decoders[5]
        else:
                data=''
                ecc=''
                bitflips=-1
                decoder=None

        if decoder:
                bitflips=decoder.ECCstate.t

        return (bitflips,data,ecc,decoder,wm_version)  # unsupported or corrupt wmark



    def encode_text(self, text: List[str]):
        return np.array([self._encode_text(t) for t in text])

    def encode_binary(self, text: List[str]):
        return np.array([self._encode_binary(t) for t in text])

    def _encode_binary(self, strbin):
        return self.process_encode(str(strbin))

    def _encode_text(self, text: str):
        data = self.encode_text_ascii(text)  # bytearray
        packet_d = ''.join(format(x, '08b') for x in data)
        return self.process_encode(packet_d)

    def process_encode(self,packet_d):
        data_bitcount=self.payload_len-self.bch_encoder.get_ecc_bits()-self.versionbits
        ecc_bitcount=self.bch_encoder.get_ecc_bits()

        packet_d=packet_d[0:data_bitcount]
        packet_d = packet_d+'0'*(data_bitcount-len(packet_d))

        if (len(packet_d)%8)==0:
           pad_d=0
        else:
           pad_d=8-len(packet_d)% 8
        paddedpacket_d = packet_d + ('0'*pad_d)
        padded_data = bytearray(bytes(int(paddedpacket_d[i: i + 8], 2) for i in range(0, len(paddedpacket_d), 8)))
 
        ecc = self.bch_encoder.encode(padded_data)  

        packet_e = ''.join(format(x, '08b') for x in ecc)
        packet_e = packet_e[0:ecc_bitcount]
        if (len(packet_e)%8)==0 or not (self.encoding_mode==0):
           pad_e=0
        else:
           pad_e=8-len(packet_e)% 8
        packet_e = packet_e + ('0'*pad_e)

        version=self.encoding_mode
        packet_v = ''.join(format(version, '04b'))
        packet = packet_d + packet_e + packet_v
        packet = [int(x) for x in packet]
        assert self.payload_len == len(packet),f'Error! Could not form complete packet'
        packet = np.array(packet, dtype=np.float32)
        return packet
    
    def decode_bitstream(self, data: np.array, MODE='text'):
        assert len(data.shape)==2
        return [self._decode_text(d, MODE) for d in data]



    def decode_text(self, data: np.array):
        assert len(data.shape)==2
        return [self._decode_text(d) for d in data]


    def decode_binary(self, data: np.array):
        assert len(data.shape)==2
        return [self._decode_text(d) for d in data]

    
    def _decode_text(self, packet: np.array, MODE):
        assert len(packet.shape)==1
        bitflips, packet_d, packet_e, bch_decoder, version = self.raw_payload_split(packet)
        if (bitflips==-1): # unsupported or corrupt wm
            return '', False, version
        if (len(packet_d)%8 ==0):
           pad_d=0
        else:
           pad_d=8-len(packet_d)% 8
        if (len(packet_e)%8 ==0):
           pad_e=0
        else:
           pad_e=8-len(packet_e)% 8
        packet_d = packet_d + ('0'*pad_d)
        packet_e = packet_e + ('0'*pad_e)

        packet_d = bytes(int(packet_d[i: i + 8], 2) for i in range(0, len(packet_d), 8))
        packet_e = bytes(int(packet_e[i: i + 8], 2) for i in range(0, len(packet_e), 8))
        data = bytearray(packet_d)
        ecc = bytearray(packet_e)
        data0 = self.decode_text_ascii(deepcopy(data)).rstrip('\x00').strip()
        if len(ecc)==bch_decoder.get_ecc_bytes():
            bitflips = bch_decoder.decode(data, ecc) 
        else:
            bitflips = -1 
#        print('Bitflips = %d' % bitflips)
        if bitflips == -1:
            if MODE=='text':
               data = data0
               return data, False, version
            else:
               dataasc = ''.join(format(x, '08b') for x in data)
               maxbits=self.schemaCapacity(version)
               dataasc=dataasc[0:maxbits]
               return dataasc, False, version

        else:
            if MODE=='text':
                dataasc = self.decode_text_ascii(data).rstrip('\x00').strip()
            else:
                dataasc = ''.join(format(x, '08b') for x in data)
                maxbits=self.schemaCapacity(version)
                dataasc=dataasc[0:maxbits]
            return dataasc, True, version


    def encode_text_ascii(self, text: str):
        # encode text to 7-bit ascii
        # input: text, str
        # output: encoded text, bytearray
        text_int7 = [ord(t) & 127 for t in text]
        text_bitstr = ''.join(format(t,'07b') for t in text_int7)
        if len(text_bitstr) % 8 != 0:
            text_bitstr =  text_bitstr + '0'*(8-len(text_bitstr)%8) # + text_bitstr  # pad to multiple of 8
        text_int8 = [int(text_bitstr[i:i+8], 2) for i in range(0, len(text_bitstr), 8)]
        return bytearray(text_int8)


    def decode_text_ascii(self, text: bytearray):
        # decode text from 7-bit ascii
        # input: text, bytearray
        # output: decoded text, str
        text_bitstr = ''.join(format(t,'08b') for t in text)  # bit string
        text_int7 = [int(text_bitstr[i:i+7], 2) for i in range(0, len(text_bitstr), 7)]
        text_bytes = bytes(text_int7)
        return text_bytes.decode('utf-8')



## TESTING ONLY

# Copyright 2023 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

def random_string(string_length=7):
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.


def main():

   # test rig for error correction
   N=100 # nreps
   bitflips = 5
   mode = 4
   for i in range(0,N):
      # mode=random.randint(0, 3) 
      D=DataLayer(100,False,mode)
      nbit=D.schemaCapacity(mode)
      string_secret=''.join([random.choice(['0', '1']) for _ in range(nbit)])
      wmpayload = D.encode_binary([string_secret])   
      watermark = torch.from_numpy(wmpayload)
      print(watermark.shape)
      # bitflips= random.randint(0,D.bch_encoder.ECCstate.t)   
      CORRUPTBITS=[random.randint(0, len(wmpayload[0])-5) for _ in range(bitflips)]
      for b in CORRUPTBITS:
         wmpayload[0][b] = 1- wmpayload[0][b]
      secret_pred, detected, version = D.decode_bitstream(wmpayload,MODE='binary')[0]
   #  print("Recovered - orig[%s] recover[%s] (det=%s schema %d bitflips=%d)" % (string_secret,secret_pred,detected,version,bitflips))
      # print("(det=%s schema %d bitflips=%d)" % (detected,version,bitflips))
      print(type(secret_pred))
      for c in range(0,len(string_secret)): 
         if string_secret[c]!=secret_pred[c]:
            print("STOP - MISMATCH!")
            quit()

if __name__ == "__main__":
   main()