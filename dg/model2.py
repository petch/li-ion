from dolfin import *
import math

R = 8.3144621   # Gas constant J/K/mol
T = 300.        # Temperature K
F = 96485.33    # Faraday constant A*s/mol
RTF = R*T/F

ke = 2.0e-3     # Electrical conductivity of electrolyte S/cm
De = 7.5e-7     # Interdiffusion coefficient of electrolyte cm2/s
te = 0.363      # Transference number of Li ions in electrolyte -

cam = 0.024681  # Maximum concentration in anode mol/cm3
ka = 1.         # Electrical conductivity of anode S/cm
Da = 3.9e-10    # Interdiffusion coefficient of anode cm2/s
ra = 0.002      # Reaction rate of anode A*cm2.5/mol1.5
def Ua(soc):    # Half cell open circuit potential of anode V
    return -0.132 + exp(-3.52*soc)
def dUa(soc):
    return -3.52*exp(-3.52*soc)

ccm = 0.023671  # Maximum concentration in cathode mol/cm3
kc = 0.038      # Electrical conductivity of cathode S/cm
Dc = 1e-9       # Interdiffusion coefficient of cathode cm2/s
rc = 0.2        # Reaction rate of cathode A*cm2.5/mol1.5
def Uc(soc):    # Half cell open circuit potential of cathode V
    return 4.06279 \
         + 0.0677504*tanh(-21.8502*soc + 12.8268) \
         - 0.045*exp(-71.69*pow(soc, 8)) \
         - 0.105734*(pow(1.00167 - soc, -0.379571) - 1.576) \
         + 0.01*exp(-200*(soc-0.19))
def dUc(soc):
    return -0.0677504*21.8502/pow(cosh(-21.8502*soc + 12.8268), 2) \
         + 0.045*71.69*8*exp(-71.69*pow(soc, 8))*pow(soc, 7) \
         - 0.105734*0.379571*pow(1.00167 - soc, -1.379571) \
         - 0.01*200*exp(-200*(soc-0.19))

def C1(Vc, Gc):                 # 1C rate A/cm2
    return ccm*F*Vc/(Gc*3600)
def Ca(ca0, dt, g, Gc, Vc):     # Constant concentration in anode mol/cm3
    return ca0 - dt*g/F*Gc/Vc
def Cc(cc0, dt, g, Gc, Vc):     # Constant concentration in anode mol/cm3
    return cc0 + dt*g/F*Gc/Vc
def Pa(c):                      # Constant potential in anode V
    return Ua(c/cam)
def Pe(Ia, Gc, g, ce, c, p):    # Constant potential in electrolyte V
    ash = math.asinh(Gc*g/(2*Ia*ra*sqrt(ce*c*(cam-c))))
    return p - 2*ash*RTF - Ua(c/cam)
def Pc(Ic, Gc, g, c, ce, pe):   # Constant potential in cathode V
    ash = math.asinh(Gc*g/(2*Ic*rc*sqrt(ce*c*(ccm-c))))
    return pe - 2*ash*RTF + Uc(c/ccm)

def ja(c, p):   # Electrical current in anode
    return -ka*grad(p)
def Na(c, p):   # Concentration flux in anode
    return -Da*grad(c)
def dja(c, p, dc, dp):
    return ja(dc, dp)
def dNa(c, p, dc, dp):
    return Na(dc, dp)

def jc(c, p):   # Electrical current in cathode
    return -kc*grad(p)
def Nc(c, p):   # Concentration flux in cathode
    return -Dc*grad(c)
def djc(c, p, dc, dp):
    return jc(dc, dp)
def dNc(c, p, dc, dp):
    return Nc(dc, dp)

def je(c, p):   # Electrical current in electrolyte
    return -ke*grad(p) + ke*(1-te)*R*T/F/c*grad(c)
def Ne(c, p):   # Concentration flux in electrolyte
    return -De*grad(c) + te/F*je(c, p)
def dje(c, p, dc, dp):
    return -ke*grad(dp) + ke*(1-te)*R*T/F/c*(grad(dc) - dc/c*grad(c))
def dNe(c, p, dc, dp):
    return -De*grad(dc) + te/F*dje(c, p, dc, dp)

def Nua(c, p, pe):
    return p - pe - Ua(c/cam)
def ia(c, ce, p, pe):
    return 2*ra*sqrt(ce*c*(cam-c))*sinh(Nua(c, p, pe)/RTF/2)
def diapa(c, ce, p, pe):
    return ra*sqrt(ce*c*(cam-c))*cosh(Nua(c, p, pe)/RTF/2)/RTF
def diaca(c, ce, p, pe):
    return ra*sqrt(ce/(c*(cam-c)))*(cam-2*c)*sinh(Nua(c, p, pe)/RTF/2) - diapa(c, ce, p, pe)*dUa(c/cam)/cam
def diape(c, ce, p, pe):
    return ra*sqrt(ce*c*(cam-c))*cosh(Nua(c, p, pe)/RTF/2)/RTF
def diace(c, ce, p, pe):
    return -ra*sqrt(c*(cam-c)/ce)*sinh(Nua(c, p, pe)/RTF/2)

def Nuc(c, p, pe):
    return p - pe - Uc(c/ccm)
def ic(c, ce, p, pe):
    return 2*rc*sqrt(ce*c*(ccm-c))*sinh(Nuc(c, p, pe)/RTF/2)
def dicpc(c, ce, p, pe):
    return rc*sqrt(ce*c*(ccm-c))*cosh(Nuc(c, p, pe)/RTF/2)/RTF
def diccc(c, ce, p, pe):
    return rc*sqrt(ce/(c*(ccm-c)))*(ccm-2*c)*sinh(Nuc(c, p, pe)/RTF/2) - dicpc(c, ce, p, pe)*dUc(c/ccm)/ccm
def dicpe(c, ce, p, pe):
    return rc*sqrt(ce*c*(ccm-c))*cosh(Nuc(c, p, pe)/RTF/2)/RTF
def dicce(c, ce, p, pe):
    return -rc*sqrt(c*(ccm-c)/ce)*sinh(Nuc(c, p, pe)/RTF/2)
